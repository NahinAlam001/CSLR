import os
import yaml
import logging
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from src.data import SignLanguageDataset, pad_collate_fn, train_tokenizer
from src.models import TransformerEncoder, TransformerDecoder, Seq2Seq
from src.evaluate import evaluate_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
writer = SummaryWriter('runs/cslr_experiment')

def set_seeds(seed: int = 42):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    set_seeds()
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    os.makedirs('processed_data', exist_ok=True)
    tokenizer_path = config['tokenizer_path']
    train_tsv = os.path.join(config['tsv_dir'], 'cvpr23.fairseq.i3d.train.how2sign.tsv')
    if not os.path.exists(tokenizer_path):
        train_tokenizer(train_tsv, tokenizer_path, config['vocab_size'])
    tokenizer = Tokenizer.from_file(tokenizer_path)

    train_dataset = SignLanguageDataset(train_tsv, config['feature_dir'], tokenizer, 
                                        config['max_seq_len'], 'train', config['augment_prob'])
    val_tsv = os.path.join(config['tsv_dir'], 'cvpr23.fairseq.i3d.val.how2sign.tsv')
    val_dataset = SignLanguageDataset(val_tsv, config['feature_dir'], tokenizer, config['max_seq_len'], 'val')
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, 
                              collate_fn=pad_collate_fn, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, 
                            collate_fn=pad_collate_fn, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    enc = TransformerEncoder(config['input_dim'], config['hid_dim'], config['n_layers'], 
                             config['n_heads'], config['pf_dim'], config['dropout'])
    dec = TransformerDecoder(config['output_dim'], config['hid_dim'], config['n_layers'], 
                             config['n_heads'], config['pf_dim'], config['dropout'])
    model = Seq2Seq(enc, dec, device, tokenizer.token_to_id("[SOS]"), 
                    tokenizer.token_to_id("[EOS]"), tokenizer.token_to_id("[PAD]")).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'])
    scheduler = CosineAnnealingLR(optimizer, T_max=config['num_epochs'])
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id("[PAD]"))

    best_val_loss = float('inf')
    patience_counter = 0
    for epoch in range(config['num_epochs']):
        model.train()
        epoch_loss = 0
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            src, trg = batch
            if src.nelement() == 0: continue
            src, trg = src.to(device), trg.to(device)
            optimizer.zero_grad()
            output = model(src, trg)
            output_dim = output.shape[-1]
            output = output.reshape(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)
            loss = criterion(output, trg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            writer.add_scalar('Train/Loss', loss.item(), epoch * len(train_loader) + len(train_loader))

        train_loss = epoch_loss / len(train_loader)
        val_loss, val_bleu, val_rouge, val_wer = evaluate_model(model, val_loader, criterion, tokenizer, device)
        writer.add_scalar('Val/Loss', val_loss, epoch)
        writer.add_scalar('Val/BLEU', val_bleu, epoch)
        writer.add_scalar('Val/ROUGE', val_rouge, epoch)
        writer.add_scalar('Val/WER', val_wer, epoch)
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), config['model_save_path'])
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                logging.info("Early stopping triggered.")
                break

        logging.info(f"Epoch {epoch+1}: Train Loss {train_loss:.3f} | Val Loss {val_loss:.3f} | Val BLEU {val_bleu:.2f} | Val ROUGE {val_rouge:.2f} | Val WER {val_wer:.2f}%")

if __name__ == "__main__":
    main()
