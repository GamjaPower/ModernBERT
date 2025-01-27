import os
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file('./work/fineweb2_tokenizer/tokenizer.json')

test_sentence = "안녕하세요 테스트 문장입니다."

# 토큰화 실행
encoding = tokenizer.encode(test_sentence)

# 검증
assert len(encoding.tokens) > 0, "토큰화 결과가 비어있습니다."
assert isinstance(encoding.tokens, list), "토큰 결과가 리스트 형태가 아닙니다."

print(encoding.tokens)