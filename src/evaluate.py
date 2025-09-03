import torch
import evaluate
import logging
from tqdm import tqdm
from tokenizers import Tokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def compute_wer(predictions: list, references: list) -> float:
    """Custom Word Error Rate computation."""
    wer = 0
    for pred, ref in zip(predictions, references):
        pred_words = pred.split()
        ref_words = ref[0].split()
        if len(ref_words) == 0:
            wer += 1
        else:
            wer += len(set(pred_words) ^ set(ref_words)) / len(ref_words)
    return wer / len(predictions) * 100 if predictions else 0

def evaluate_model(model: torch.nn.Module, iterator: torch.utils.data.DataLoader, 
                   criterion: torch.nn.Module, tokenizer: Tokenizer, device: torch.device):
    model.eval()
    epoch_loss = 0
    bleu_metric = evaluate.load("sacrebleu")
    rouge_metric = evaluate.load("rouge")
    predictions = []
    references = []
    with torch.no_grad():
        for batch in tqdm(iterator, desc="Evaluating"):
            src, trg = batch
            if src.nelement() == 0: continue
            src, trg = src.to(device), trg.to(device)
            output = model(src, trg)
            output_dim = output.shape[-1]
            flat_output = output.reshape(-1, output_dim)
            flat_trg = trg[:, 1:].reshape(-1)
            loss = criterion(flat_output, flat_trg)
            epoch_loss += loss.item()

            # Use generate for predictions
            predicted_indices = model.generate(src)
            decoded_preds = tokenizer.decode_batch(predicted_indices.cpu().numpy())
            decoded_refs = tokenizer.decode_batch(trg.cpu().numpy())
            cleaned_preds = [p.replace("[PAD]", "").replace("[SOS]", "").replace("[EOS]", "").strip() for p in decoded_preds]
            cleaned_refs = [[r.replace("[PAD]", "").replace("[SOS]", "").replace("[EOS]", "").strip()] for r in decoded_refs]
            predictions.extend(cleaned_preds)
            references.extend(cleaned_refs)

    predictions = [pred if pred else " " for pred in predictions]
    bleu_score = bleu_metric.compute(predictions=predictions, references=references)['score']
    rouge_score = rouge_metric.compute(predictions=predictions, references=references)['rougeL'] * 100
    wer_score = compute_wer(predictions, references)
    return epoch_loss / len(iterator), bleu_score, rouge_score, wer_score

def main():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    tokenizer = Tokenizer.from_file(config['tokenizer_path'])
    test_tsv = os.path.join(config['tsv_dir'], 'cvpr23.fairseq.i3d.test.how2sign.tsv')
    test_dataset = SignLanguageDataset(test_tsv, config['feature_dir'], tokenizer, config['max_seq_len'], 'test')
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=pad_collate_fn, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    enc = TransformerEncoder(config['input_dim'], config['hid_dim'], config['n_layers'], 
                             config['n_heads'], config['pf_dim'], config['dropout'])
    dec = TransformerDecoder(config['output_dim'], config['hid_dim'], config['n_layers'], 
                             config['n_heads'], config['pf_dim'], config['dropout'])
    model = Seq2Seq(enc, dec, device, tokenizer.token_to_id("[SOS]"), 
                    tokenizer.token_to_id("[EOS]"), tokenizer.token_to_id("[PAD]")).to(device)
    model.load_state_dict(torch.load(config['model_save_path'], map_location=device))

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id("[PAD]"))
    test_loss, test_bleu, test_rouge, test_wer = evaluate_model(model, test_loader, criterion, tokenizer, device)
    logging.info("--- Evaluation Complete ---")
    logging.info(f"Test Loss: {test_loss:.3f}")
    logging.info(f"Test BLEU Score: {test_bleu:.2f}")
    logging.info(f"Test ROUGE Score: {test_rouge:.2f}")
    logging.info(f"Test WER: {test_wer:.2f}%")

if __name__ == "__main__":
    main()
