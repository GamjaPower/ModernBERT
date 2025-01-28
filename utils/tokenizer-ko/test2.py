import os
from tokenizers import Tokenizer
from transformers import AutoTokenizer

# tokenizer = Tokenizer.from_file('./work/fineweb2_tokenizer/tokenizer.json')
# tokenizer = Tokenizer.from_file('./work/fineweb2_100_tokenizer/tokenizer.json')
# tokenizer = Tokenizer.from_file('./work/fineweb2_pure_tokenizer/tokenizer.json')
tokenizer = AutoTokenizer.from_pretrained("./work/fineweb_hand_tokenizer")


test_sentence = "안녕하세요 테스트 문장입니다."

print(tokenizer.tokenize(test_sentence))

# 토큰화 실행
# encoding = tokenizer.encode(test_sentence)

# print(encoding.tokens)