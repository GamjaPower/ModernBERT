from transformers import pipeline
from pprint import pprint
import torch
import os

def main():

    pipe = pipeline(
        "fill-mask",
        # model="answerdotai/ModernBERT-base",
        model="./models/modernbert-base-kr",
        # model="./models/modernbert-base-uncased",
        # torch_dtype=torch.float32,
        # pipeline_class='ModernBertForMaskedLM',
    )

    # input_text = "He walked to the [MASK]."
    # input_text = "She doned to the [MASK]."
    input_text = "다음주 월요일에 온라인 [MASK]."
    # input_text = "안철수를 떠나며 [MASK] 말했다는 것이 다시 주목받게 됐다. "
    results = pipe(input_text)
    pprint(results)

if __name__ == "__main__":
    main()