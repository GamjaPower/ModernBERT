import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset
from sklearn.metrics import accuracy_score
import numpy as np

# 데이터셋 로드
dataset = load_dataset("Blpeng/nsmc")

# BERT 토크나이저 설정
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

# 데이터 전처리 함수
def preprocess_function(examples):
    return tokenizer(examples["document"], truncation=True, padding="max_length", max_length=128)

# 데이터셋 전처리
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# 평가 함수 정의
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {"accuracy": accuracy_score(labels, predictions)}

# 모델 설정
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-multilingual-cased", 
    num_labels=2
)

# 학습 인자 설정
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True
)

# 트레이너 초기화
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    compute_metrics=compute_metrics,
)

# 모델 학습
trainer.train()

# 추론 예시
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding=True)
    outputs = model(**inputs)
    prediction = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return "긍정" if prediction[0][1] > prediction[0][0] else "부정"
