from composer.models import HuggingFaceModel
from transformers import AutoTokenizer

def convert():
    # Composer 체크포인트에서 모델과 토크나이저 로드
    model, tokenizer = HuggingFaceModel.hf_from_composer_checkpoint(
        checkpoint_path='./work/ep2916-ba70000-rank0.pt',
        model_instantiation_class='transformers.AutoModelForSequenceClassification', 
        model_config_kwargs={'num_labels': 2}  # 필요한 설정 추가
    )
    bert_tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-uncased')

    model.save_pretrained('./work/hf_model')
    bert_tokenizer.save_pretrained('./work/hf_model')
    


if __name__ == "__main__":
    convert()