
import sentencepiece as spm

# 모델 로딩
sp = spm.SentencePieceProcessor()
# sp.load('./work/fineweb2_spm_tokenizer/spm_mdeberta.model')
sp.load('./models/modernbert-base-kr/spm.model')


test_sentence = "안녕하세요 테스트 문장입니다."

# 텍스트를 토큰화
tokens = sp.encode(test_sentence, out_type=str)

print(tokens)
