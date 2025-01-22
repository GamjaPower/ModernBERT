from transformers import BertTokenizer, AutoTokenizer, ModernBertForMaskedLM
from transformers import pipeline
from pprint import pprint
import torch
import os

def main():

    pipe = pipeline(
        "fill-mask",
        # model="answerdotai/ModernBERT-base",
        model="./models/modernbert-base-en",
        # model="./models/modernbert-base-uncased",
        # pipeline_class='ModernBertForMaskedLM',
    )

    # input_text = "She walked to the [MASK]."
    # input_text = "The Denver Board of Education [MASK]."
    input_text = "David Ortiz finished the best April of his [MASK]"
    results = pipe(input_text)
    pprint(results)


if __name__ == "__main__":
    main()
    # token()