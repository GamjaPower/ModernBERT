import os
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file('./work/fineweb2_10000_tokenizer/tokenizer.json')
# tokenizer = Tokenizer.from_file('./models/modernbert-base-en/tokenizer.json')

test_sentence = "안녕하세요 테스트 문장입니다."

# 토큰화 실행
encoding = tokenizer.encode(test_sentence)


print(encoding.tokens)