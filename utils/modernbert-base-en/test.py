from transformers import BertTokenizer, AutoTokenizer, ModernBertForMaskedLM
from transformers import pipeline
from pprint import pprint
from composer.models import write_huggingface_pretrained_from_composer_checkpoint

import torch
import os

def main():

    pipe = pipeline(
        "fill-mask",
        model="./models/modernbert-base-en",
        torch_dtype=torch.float32,
    )

    input_text = "She walked to the [MASK]."
    # input_text = "The Denver Board of Education [MASK]."
    # input_text = "David Ortiz finished the best April of his [MASK]"
    results = pipe(input_text)
    pprint(results)


if __name__ == "__main__":
    main()
    # token()