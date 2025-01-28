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
NUM_EXAMPLES_TO_TRAIN = 1  # Number of examples to use from the streaming dataset
BATCH_SIZE = 1000

# --- Tokenizer Training ---

def train_tokenizer():

    old_tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")

    # 데이터셋 로드 (예시)
    # dataset = load_dataset("stanfordnlp/imdb", split='train')
    # dataset = load_dataset("maywell/korean_textbooks", 'claude_evol',  split='train')
    dataset = load_dataset(DATASET_NAME, SUB_DATASET_NAME, split='train')

    # 배치 이터레이터 생성
    def batch_iterator(dataset, batch_size=1000):
        for i in range(0, NUM_EXAMPLES_TO_TRAIN, 1): # len(dataset)
            # yield dataset[i:i+batch_size]["text"]
            yield('')


    tokens_dict = { 
        'unk_token': '[UNK]',
        'sep_token': '[SEP]',
        'pad_token': '[PAD]',
        'cls_token': '[CLS]',
        'mask_token': '[MASK]',
    }
    new_token_spaces = [
        '                        ',
        '                       ',
        '                      ',
        '                     ',
        '                    ',
        '                   ',
        '                  ',
        '                 ',
        '                ',
        '               ',
        '              ',
        '             ',
        '            ',
        '           ',
        '          ',
        '         ',
        '        ',
        '       ',
        '      ',
        '     ',
        '    ',
        '   ',
        '  ',    
    ]
    
    # old_tokenizer.add_special_tokens(special_tokens_dict = tokens_dict, replace_additional_special_tokens=True)
    new_tokenizer = old_tokenizer.train_new_from_iterator(
        batch_iterator(dataset),
        vocab_size=len(old_tokenizer.get_vocab())-26, # 3 + 23
        min_frequency=2,
    )
    new_tokens = [ 
        AddedToken('|||IP_ADDRESS|||', rstrip=False, lstrip=False, normalized=True, special=False),
        AddedToken('<|padding|>', lstrip=False, rstrip=False, normalized=False, special=True),
        AddedToken('<|endoftext|>', lstrip=False, rstrip=False, normalized=False, special=True),
        AddedToken('|||EMAIL_ADDRESS|||', rstrip=False, lstrip=False, normalized=True, special=False),
        AddedToken('|||PHONE_NUMBER|||', rstrip=False, lstrip=False, normalized=True, special=False),
    ]
 
    for new_token_space in new_token_spaces:
        new_tokens.append(AddedToken(new_token_space, rstrip=False, lstrip=False, normalized=True, special=False))
    
    new_tokenizer.add_tokens(new_tokens, special_tokens=False)
    new_tokenizer.save_pretrained("./work/fineweb2_100_tokenizer")
    

if __name__ == "__main__":
    train_tokenizer()
