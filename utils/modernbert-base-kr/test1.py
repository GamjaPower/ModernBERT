from transformers import pipeline, AutoTokenizer
from pprint import pprint
import torch
import os

def main():

    # tokenizer = AutoTokenizer.from_pretrained("./models/modernbert-base-en")
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    
    

    test_sentence = "Hi."

    # 토큰화 실행
    encoding = tokenizer.encode(test_sentence)
    print(encoding)

if __name__ == "__main__":
    main()