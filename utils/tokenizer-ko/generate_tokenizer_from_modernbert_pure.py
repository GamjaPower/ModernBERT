from transformers import AutoTokenizer
from tokenizers import Tokenizer, AddedToken
from tokenizers.models import WordPiece
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import WordPieceTrainer
from tokenizers.normalizers import NFC
from datasets import load_dataset
from itertools import islice
import os

# --- Configuration ---
DATASET_NAME = "HuggingFaceFW/fineweb-2"  # Dataset for tokenizer training
SUB_DATASET_NAME = "kor_Hang"  # Sub-dataset for tokenizer training
NUM_EXAMPLES_TO_TRAIN = 10  # Number of examples to use from the streaming dataset
BATCH_SIZE = 1000

# --- Tokenizer Training ---

def train_tokenizer():

    old_tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")

    # 데이터셋 로드 (예시)
    dataset = load_dataset(DATASET_NAME, SUB_DATASET_NAME, split='train')

    # 배치 이터레이터 생성
    def batch_iterator(dataset, batch_size=BATCH_SIZE):
        for i in range(0, NUM_EXAMPLES_TO_TRAIN, BATCH_SIZE): # len(dataset)
            # yield dataset[i:i+batch_size]["text"]
            yield('')

    new_tokenizer = old_tokenizer.train_new_from_iterator(
        batch_iterator(dataset),
        vocab_size=len(old_tokenizer.get_vocab()),
        min_frequency=2,
    )
 
    new_tokenizer.save_pretrained("./work/fineweb2_pure_tokenizer")
    

if __name__ == "__main__":
    train_tokenizer()
