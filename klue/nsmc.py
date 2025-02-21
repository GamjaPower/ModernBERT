from datasets import load_dataset, ClassLabel
from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os
import argparse


def eval():

    # # ArgumentParser 객체 생성
    # parser = argparse.ArgumentParser(description='명령줄 인자 예제')    
    # # 인자 추가
    # parser.add_argument('--model', type=str, help='model name', required=True)
    # args = parser.parse_args()
    # model_name = 'lighthouse/mdeberta-v3-base-kor-further'
    model_name = './models/modernbert-base-kr'
    
    # 데이터셋 로드 (Naive Sentiment Movie Corpus)
    # dataset = load_dataset('Blpeng/nsmc')
    dataset = load_dataset('Bingsu/namuwiki_20210301_filtered')
    
    # dataset = load_dataset('klue/klue', 'ynat')


    

    # 토크나이저와 모델 로드
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2
    )

    # 데이터 전처리 함수
    def preprocess_function(examples):
        texts = [str(doc) if doc is not None else "" for doc in examples['document']]
        return tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=128
        )
    

    # 데이터셋 전처리
    tokenized_dataset = dataset.map(preprocess_function, batched=True)

    # 평가 메트릭 함수
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, 
            preds, 
            average='macro'
        )
        acc = accuracy_score(labels, preds)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

    # 학습 파라미터 설정
    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy='epoch',
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
    )
    

    # Trainer 초기화
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['test'],
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # 평가 실행
    evaluation_results = trainer.evaluate()

    
    print(evaluation_results)


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    eval()