import os
import random
import math
import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class KeypointAugment:
    """Augmentation for keypoints (MediaPipe features).
    
    Applies rotation, scaling, translation with given probability.
    """
    def __init__(self, prob: float = 0.5):
        self.prob = prob

    def __call__(self, features: np.ndarray) -> np.ndarray:
        if random.random() > self.prob:
            return features

        T, D = features.shape
        mediapipe_flat = features[:, :99]
        i3d = features[:, 99:]

        mediapipe = mediapipe_flat.reshape(T, 33, 3)

        # Rotation
        angle = random.uniform(-15, 15) * math.pi / 180
        cos, sin = math.cos(angle), math.sin(angle)
        rot_matrix = np.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]])
        mediapipe = np.einsum('tnd,dd->tnd', mediapipe, rot_matrix)

        # Scaling
        scale = random.uniform(0.9, 1.1)
        mediapipe *= scale

        # Translation
        trans = np.random.uniform(-0.1, 0.1, size=(1, 1, 3))
        mediapipe += trans

        mediapipe_flat = mediapipe.reshape(T, 99)
        return np.concatenate([mediapipe_flat, i3d], axis=1)

class SignLanguageDataset(Dataset):
    """Dataset for sign language features and translations."""
    def __init__(self, tsv_file: str, feature_base_dir: str, tokenizer: Tokenizer, 
                 max_seq_len: int = 100, split: str = 'train', augment_prob: float = 0.0):
        self.annotations = pd.read_csv(tsv_file, sep='\t')
        self.mediapipe_dir = os.path.join(feature_base_dir, 'mediapipe_features_how2sign/mediapipe_features', split)
        self.i3d_dir = os.path.join(feature_base_dir, 'i3d_features_how2sign/i3d_features_how2sign', split)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.pad_token_id = tokenizer.token_to_id("[PAD]")
        self.sos_token_id = tokenizer.token_to_id("[SOS]")
        self.eos_token_id = tokenizer.token_to_id("[EOS]")
        self.augment = KeypointAugment(prob=augment_prob)

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> tuple:
        row = self.annotations.iloc[idx]
        video_id = row['id']
        sentence = row['translation']
        mediapipe_path = os.path.join(self.mediapipe_dir, f"{video_id}.npy")
        i3d_path = os.path.join(self.i3d_dir, f"{video_id}.npy")

        if not os.path.exists(mediapipe_path) or not os.path.exists(i3d_path):
            logging.warning(f"Feature files not found for {video_id}. Skipping.")
            return None, None

        mediapipe = np.load(mediapipe_path)
        i3d = np.load(i3d_path)

        min_t = min(mediapipe.shape[0], i3d.shape[0])
        mediapipe = mediapipe[:min_t]
        i3d = i3d[:min_t]

        mediapipe_flat = mediapipe.reshape(min_t, -1)
        features = np.concatenate([mediapipe_flat, i3d], axis=1)
        features = self.augment(features)

        tokenized = self.tokenizer.encode(str(sentence))
        token_ids = [self.sos_token_id] + tokenized.ids + [self.eos_token_id]
        padded_tokens = token_ids + [self.pad_token_id] * (self.max_seq_len - len(token_ids))
        padded_tokens = padded_tokens[:self.max_seq_len]

        return torch.FloatTensor(features), torch.LongTensor(padded_tokens)

def pad_collate_fn(batch):
    batch = [b for b in batch if b[0] is not None]
    if not batch:
        return torch.tensor([]), torch.tensor([])
    features, tokens = zip(*batch)
    features_padded = pad_sequence(features, batch_first=True, padding_value=0.0)
    tokens_stacked = torch.stack(tokens)
    return features_padded, tokens_stacked

def train_tokenizer(tsv_file: str, output_path: str, vocab_size: int = 5000):
    """Train BPE tokenizer on translations."""
    annotations = pd.read_csv(tsv_file, sep='\t')
    sentences = annotations['translation'].tolist()
    logging.info(f"Loaded {len(sentences)} sentences for tokenizer training.")

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"])
    tokenizer.train_from_iterator(sentences, trainer=trainer)
    tokenizer.save(output_path)
    logging.info(f"Tokenizer trained with vocab size {tokenizer.get_vocab_size()} and saved to {output_path}")
    return tokenizer
