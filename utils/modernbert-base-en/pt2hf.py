from composer.models import HuggingFaceModel
from transformers import AutoTokenizer, ModernBertForMaskedLM
from transformers import pipeline
from pprint import pprint

def convert():

    model, tokenizer = HuggingFaceModel.hf_from_composer_checkpoint(
        # checkpoint_path='./checkpoints/modernbert-base-en/ep53-ba1200-rank0.pt',
        # checkpoint_path='./checkpoints/modernbert-base-en/ep91-ba1600-rank0.pt',
        # checkpoint_path='./checkpoints/modernbert-base-en/ep265-ba2800-rank0.pt',
        checkpoint_path='./checkpoints/modernbert-base-en/ep0-ba1400-rank0.pt',
        model_instantiation_class='transformers.ModernBertForMaskedLM', 
        # model_instantiation_class='transformers.BertForMaskedLM', 

        model_config_kwargs={
            'num_labels': 2,
            "model_type": "modernbert",
        }
    )

    model.save_pretrained('./models/modernbert-base-en')
    tokenizer.save_pretrained('./models/modernbert-base-en')

    pipe = pipeline(
        "fill-mask",
        model=model,
        tokenizer=tokenizer
    )

    # input_text = "She walked to the [MASK]."
    input_text = "The Denver Board of Education [MASK]."
    # input_text = "She logged to the [MASK]."
    results = pipe(input_text)
    pprint(results)

if __name__ == "__main__":
    convert()