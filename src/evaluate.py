import os
import yaml # Fixes the NameError
import argparse # Needed for arguments
import torch
import torch.nn as nn
import evaluate
import logging
import jiwer
from tqdm import tqdm
from tokenizers import Tokenizer
from torch.utils.data import DataLoader

# Important: We need to import from other src files
from src.data import SignLanguageDataset, pad_collate_fn
from src.models import TransformerEncoder, TransformerDecoder, Seq2Seq

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def evaluate_model(model: torch.nn.Module, iterator: torch.utils.data.DataLoader,
                   criterion: torch.nn.Module, tokenizer: Tokenizer, device: torch.device, config: dict):
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

            # Handle model trained with DataParallel
            if isinstance(model, nn.DataParallel):
                predicted_indices = model.module.generate(src, max_len=config['max_seq_len'])
            else:
                predicted_indices = model.generate(src, max_len=config['max_seq_len'])

            decoded_preds = tokenizer.decode_batch(predicted_indices.cpu().numpy().tolist())
            decoded_refs = tokenizer.decode_batch(trg.cpu().numpy().tolist())

            cleaned_preds = [p.replace("[PAD]", "").replace("[SOS]", "").replace("[EOS]", "").strip() for p in decoded_preds]
            cleaned_refs_flat = [r.replace("[PAD]", "").replace("[SOS]", "").replace("[EOS]", "").strip() for r in decoded_refs]

            predictions.extend(cleaned_preds)
            references.extend(cleaned_refs_flat)

    predictions_safe = [pred if pred else " " for pred in predictions]
    references_safe = [ref if ref else " " for ref in references]

    bleu_score = bleu_metric.compute(predictions=predictions_safe, references=[[r] for r in references_safe])['score']
    rouge_score = rouge_metric.compute(predictions=predictions_safe, references=references_safe)['rougeL'] * 100
    wer_score = jiwer.wer(references_safe, predictions_safe) * 100

    return epoch_loss / len(iterator), bleu_score, rouge_score, wer_score

def main():
    parser = argparse.ArgumentParser(description="Evaluate the Sign Language Transformer")
    parser.add_argument('--feature_dir', type=str, required=True, help='Path to the downloaded features directory.')
    parser.add_argument('--tsv_dir', type=str, required=True, help='Path to the directory containing TSV files.')
    args = parser.parse_args()

    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    config['feature_dir'] = args.feature_dir
    config['tsv_dir'] = args.tsv_dir

    tokenizer = Tokenizer.from_file(config['tokenizer_path'])
    test_tsv = os.path.join(config['tsv_dir'], 'cvpr23.fairseq.i3d.test.how2sign.tsv')
    test_dataset = SignLanguageDataset(test_tsv, config['feature_dir'], tokenizer, config['max_seq_len'], 'test')
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=pad_collate_fn, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    enc = TransformerEncoder(config['input_dim'], config['hid_dim'], config['n_layers'], config['n_heads'], config['pf_dim'], config['dropout'])
    dec = TransformerDecoder(config['output_dim'], config['hid_dim'], config['n_layers'], config['n_heads'], config['pf_dim'], config['dropout'])
    model = Seq2Seq(enc, dec, device, tokenizer.token_to_id("[SOS]"), tokenizer.token_to_id("[EOS]"), tokenizer.token_to_id("[PAD]")).to(device)

    # Load the trained model weights
    state_dict = torch.load(config['model_save_path'], map_location=device)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k # remove `module.`
        new_state_dict[name] = v
    # load params
    model.load_state_dict(new_state_dict)


    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id("[PAD]"))
    test_loss, test_bleu, test_rouge, test_wer = evaluate_model(model, test_loader, criterion, tokenizer, device, config)

    logging.info("--- Evaluation Complete ---")
    logging.info(f"Test Loss: {test_loss:.3f}")
    logging.info(f"Test BLEU Score: {test_bleu:.2f}")
    logging.info(f"Test ROUGE Score: {test_rouge:.2f}")
    logging.info(f"Test WER: {test_wer:.2f}%")

if __name__ == "__main__":
    main()
