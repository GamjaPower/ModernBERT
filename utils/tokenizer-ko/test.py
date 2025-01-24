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
TOKENIZER_SAVE_PATH = "./work/fineweb_tokenizer"  # Directory to save the trained tokenizer
VOCAB_SIZE = 50368  # Desired vocabulary size
NUM_EXAMPLES_TO_TRAIN = 1000  # Number of examples to use from the streaming dataset
BATCH_SIZE = 1000

# --- Tokenizer Training ---

def train_tokenizer():

    old_tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")

    # 데이터셋 로드 (예시)
    # dataset = load_dataset("stanfordnlp/imdb", split='train')
    dataset = load_dataset("maywell/korean_textbooks", 'claude_evol',  split='train')
    # dataset = load_dataset(DATASET_NAME, SUB_DATASET_NAME, split='train')

    # 배치 이터레이터 생성
    def batch_iterator(dataset, batch_size=1000):
        for i in range(0, 10000, batch_size): # len(dataset)
            yield dataset[i:i+batch_size]["text"]

    tokens_dict = { 
        # 'bos_token': '[]',
        # 'eos_token': '[]',
        'unk_token': '[UNK]',
        'sep_token': '[SEP]',
        'pad_token': '[PAD]',
        'cls_token': '[CLS]',
        'mask_token': '[MASK]',
        'additional_special_tokens' : [
            # AddedToken('|||IP_ADDRESS|||', special=False, normalized=True),
            AddedToken('<|padding|>', lstrip=False, rstrip=False, normalized=True),
            AddedToken('<|endoftext|>', lstrip=False, rstrip=False, normalized=True),
            # '                        ',
            # '                       ',
            # '                      ',
            # '                     ',
            # '                    ',
            # '                   ',
            # '                  ',
            # '                 ',
            # '                ',
            # '               ',
            # '              ',
            # '             ',
            # '            ',
            # '           ',
            # '          ',
            # '         ',
            # '        ',
            # '       ',
            # '      ',
            # '     ',
            # '    ',
            # '   ',
            # '  ',
            # '|||EMAIL_ADDRESS|||',
            # '|||PHONE_NUMBER|||',
            '<|endoftext|>',
        ]
    }
    tokens = [AddedToken('|||IP_ADDRESS|||', rstrip=False, lstrip=False, normalized=True, special=False),]
    old_tokenizer.add_special_tokens(special_tokens_dict = tokens_dict, replace_additional_special_tokens=True)
    # old_tokenizer.added_tokens_decoder = AddedToken('<|padding|>', lstrip=False, rstrip=False, normalized=True)
    print(old_tokenizer.added_tokens_decoder)
    old_tokenizer.add_tokens(tokens, special_tokens=False)
    # old_tokenizer.add_tokens([AddedToken('|||IP_ADDRESS|||', normalized=True)], special_tokens=False)
    new_tokenizer = old_tokenizer.train_new_from_iterator(
        batch_iterator(dataset),
        vocab_size=len(old_tokenizer.get_vocab()),
        min_frequency=2,
    )

    # 새로운 토크나이저 저장
    new_tokenizer.add_tokens(tokens, special_tokens=False)
    new_tokenizer.save_pretrained("./work/new_tokenizer")
    

if __name__ == "__main__":
    train_tokenizer()
