from composer.models import HuggingFaceModel
from transformers import AutoTokenizer

# Composer 체크포인트에서 모델과 토크나이저 로드
model, tokenizer = HuggingFaceModel.hf_from_composer_checkpoint(
    checkpoint_path='checkpoints/latest-rank0.pt',
    model_instantiation_class='transformers.AutoModelForSequenceClassification', 
    model_config_kwargs={'num_labels': 2}  # 필요한 설정 추가
)
bert_tokenizer = AutoTokenizer('google-bert/bert-base-uncased')

# Hugging Face 모델로 래핑
hf_model = HuggingFaceModel(
    model=model,
    tokenizer=bert_tokenizer,
    # metrics=metrics,
    use_logits=True
)

hf_model.save_pretrained('hf_model')
