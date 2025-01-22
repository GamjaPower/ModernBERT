from transformers import BertTokenizer, BertForMaskedLM, AdamW

from transformers import pipeline
from pprint import pprint


def simple():
    save_dir = './models/bert_simple_model'
    tokenizer = BertTokenizer.from_pretrained('google-bert/bert-base-uncased')
    model = BertForMaskedLM.from_pretrained('google-bert/bert-base-uncased')
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    pipe = pipeline(
        "fill-mask",
        model=save_dir,
    )

    input_text = "She walked to the [MASK]."
    # input_text = "The Denver Board of Education [MASK]."
    # input_text = "David Ortiz finished the best April of his [MASK]"
    results = pipe(input_text)
    pprint(results)



if __name__ == "__main__":
    simple()