from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import DataCollatorForTokenClassification, TrainingArguments, Trainer
from sklearn.metrics import precision_recall_fscore_support

import numpy as np
import os
import argparse

def eval(args):

    # KLUE NER 데이터셋 로드
    dataset = load_dataset('klue/klue', 'ner')

    # 토크나이저와 모델 로드
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    labels = dataset["train"].features["ner_tags"].feature.names
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name, 
        num_labels=len(labels),
        id2label={i: label for i, label in enumerate(labels)},
        label2id={label: i for i, label in enumerate(labels)}
    )

    # 토큰화 및 레이블 정렬 함수
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples['tokens'], 
            truncation=True, 
            is_split_into_words=True
        )
        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    # 데이터셋 토큰화
    tokenized_datasets = dataset.map(
        tokenize_and_align_labels, 
        batched=True
    )

    # 데이터 콜레이터 설정
    data_collator = DataCollatorForTokenClassification(tokenizer)

    # 평가 메트릭 정의
    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [p for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [l for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        precision, recall, f1, _ = precision_recall_fscore_support(
            np.concatenate(true_labels),
            np.concatenate(true_predictions),
            average="macro"
        )
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    # 학습 인자 설정
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        logging_dir="./logs",
        num_train_epochs=1,
        weight_decay=0.01,
        report_to="none",

    )


    # 트레이너 초기화
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # 모델 학습
    trainer.train()


    # 평가 실행
    evaluation_results = trainer.evaluate()    
    print(evaluation_results)


if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = argparse.ArgumentParser(description="evaluate KLUE NER model")
    parser.add_argument("--model_name", help="model name")
    args = parser.parse_args()
    args.model_name = "./models/modernbert-base-kr"
    eval(args)