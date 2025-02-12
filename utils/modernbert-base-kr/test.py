from transformers import pipeline
from pprint import pprint
import torch
import os

def main():

    pipe = pipeline(
        "fill-mask",
        # model="answerdotai/ModernBERT-base",
        model="./models/modernbert-base-kr",
        # model="neavo/modern_bert_multilingual",
        # model="neavo/keyword_gacha_base_multilingual",
        # model="lighthouse/mdeberta-v3-base-kor-further",
        # model="./models/modernbert-base-uncased",
        # torch_dtype=torch.float32,
        # pipeline_class='ModernBertForMaskedLM',
    )

    # input_text = "He walked to the [MASK]."
    # input_text = "She doned to the [MASK]."
    # input_text = "다음주 월요일에 온라인 미팅이 [MASK]."
    input_text = "격자형 편지지에 '그리운 조선'으로 시작하는 4줄짜리 편지를 살펴보니, 북한에서 사용하는 단어는 물론 어순과 표현 모두 흉내내기에 급급했던 것 같다는 게 김 대표의 [MASK]."
    # input_text = "안철수를 떠나며 [MASK] 말했다는 것이 다시 주목받게 됐다. "
    results = pipe(input_text)
    pprint(results)

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()