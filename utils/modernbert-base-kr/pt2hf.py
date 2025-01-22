from composer.models import HuggingFaceModel
from transformers import AutoTokenizer, ModernBertForMaskedLM
from transformers import pipeline
from pprint import pprint

def convert():

    model, tokenizer = HuggingFaceModel.hf_from_composer_checkpoint(
        # checkpoint_path='./checkpoints/modernbert-base-uncased/ep9-ba1600-rank0.pt',
        checkpoint_path='./checkpoints/modernbert-base-kr/ep1-ba13000-rank0.pt',
        model_instantiation_class='transformers.ModernBertForMaskedLM', 
        
        model_config_kwargs={
            "model_type": "modernbert",
        }  # 필요한 설정 추가
    )
    model.save_pretrained('./models/modernbert-base-kr')
    tokenizer.save_pretrained('./models/modernbert-base-kr')
    

    # pipe = pipeline(
    #     "fill-mask",
    #     model="./models/modernbert-base-kr",
    # )

    # # input_text = "He walked to the [MASK]."
    # input_text = "다음주 월요일에 온라인 [MASK]"
    # # input_text = "삼성전자는 [MASK]."
    
    # results = pipe(input_text)
    # pprint(results)



if __name__ == "__main__":
    convert()