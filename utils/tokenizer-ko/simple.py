from transformers import AutoTokenizer
from datasets import load_dataset

DATASET_NAME = "stanfordnlp/imdb"  # Dataset for tokenizer training

def train_tokenizer():
    # 기존 토크나이저 로드 (예: GPT-2 토크나이저)
    old_tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")

    # 데이터셋 로드 (예시)
    # dataset = load_dataset("your_dataset_name")
    dataset = load_dataset(DATASET_NAME, split="train", streaming=False)

    # 배치 이터레이터 생성
    def batch_iterator(dataset, batch_size=1000):
        for i in range(0, len(dataset), batch_size):
            yield dataset[i:i+batch_size]["text"]

    # 새로운 토크나이저 학습
    new_tokenizer = old_tokenizer.train_new_from_iterator(
        batch_iterator(dataset),
        vocab_size=len(old_tokenizer.get_vocab())
    )

    # 새로운 토크나이저 저장
    new_tokenizer.save_pretrained("./new_tokenizer")
if __name__ == "__main__":
    train_tokenizer()