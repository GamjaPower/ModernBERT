from transformers import pipeline
from pprint import pprint
import torch
import os

def main():
    pipe = pipeline(
        "fill-mask",
        # model="answerdotai/ModernBERT-base",
        # model="google-bert/bert-base-multilingual-uncased",
        # model="microsoft/mdeberta-v3-base",
        # model="sigridjineth/ModernBERT-Korean-ColBERT-preview-v1",
        # model="kisti/korscideberta",
        model="./work/hf_model",
        torch_dtype=torch.bfloat16,
    )

    input_text = "He walked to the [MASK]."
    # input_text = "다음주 월요일에 온라인 미팅 [MASK]."
    results = pipe(input_text)
    pprint(results)

if __name__ == "__main__":
    main()