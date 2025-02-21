from transformers import pipeline, AutoTokenizer
from pprint import pprint
import torch
import os

def main():
    # tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    tokenizer = AutoTokenizer.from_pretrained("./models/modernbert-base-kr")
    # tokenizer = AutoTokenizer.from_pretrained("neavo/modern_bert_multilingual")
    

    test_sentence = "특히 영동고속도로 강릉 방향 문막휴게소에서 만종분기점까지 5km 구간에는 승용차 전용 임시 갓길차로제를 운영하기로 했다."

    tokens = tokenizer.tokenize(test_sentence)
    for x in tokens:
        print(tokenizer.convert_tokens_to_string([x]))

if __name__ == "__main__":
    main()