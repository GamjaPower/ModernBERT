## 필요한 라이브러리 설치
# pip install transformers datasets evaluate seqeval

import numpy as np
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments
)
from datasets import load_dataset
import evaluate

def eval():
    ## 1. 모델 & 토크나이저 로드
    model_name = "lighthouse/mdeberta-v3-base-kor-further"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=13,  # KLUE NER 라벨 개수
        id2label={i: f"LABEL_{i}" for i in range(13)},  # 실제 라벨 매핑 필요
        label2id={f"LABEL_{i}": i for i in range(13)}
    )

    ## 2. 데이터셋 준비
    dataset = load_dataset("klue/klue", "ner")

    ## 3. 전처리 함수
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples["tokens"],
            truncation=True,
            is_split_into_words=True,
            max_length=512
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

    tokenized_dataset = dataset.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=dataset["train"].column_names
    )

    ## 4. 평가 메트릭 설정
    seqeval = evaluate.load("seqeval")

    label_list = [f"LABEL_{i}" for i in range(13)]  # Define label_list

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = seqeval.compute(
            predictions=true_predictions,
            references=true_labels,
            mode="strict",
            scheme="IOB2"
        )
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    ## 5. 평가 실행
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="./results",
            per_device_eval_batch_size=16,
            report_to="none"
        ),
        data_collator=data_collator,
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
    )

    results = trainer.evaluate()
    print(results)


if __name__ == "__main__":
    eval()