from transformers import (
    BertTokenizer, 
    BertForTokenClassification,
    Trainer, 
    TrainingArguments
)
from datasets import load_dataset
import os
import torch

def main():
    # 데이터셋 로드
    dataset = load_dataset("datasciathlete/open-ner-english-aihub-korean")

    # 토크나이저와 모델 초기화 
    tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
    model = BertForTokenClassification.from_pretrained(
        "bert-base-uncased", 
        num_labels=9  # 데이터셋의 레이블 수에 맞게 조정
    )

    # 데이터 전처리 함수
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples["text"],
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        labels = examples["entities"]
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)

    # Trainer 설정
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()

def compute_metrics(p):
    predictions, labels = p
    predictions = torch.argmax(predictions, dim=2)
    # 필요한 경우 추가적인 평가 지표 계산
    return {}

if __name__ == "__main__":
    main()