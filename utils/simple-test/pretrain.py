import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForMaskedLM, AdamW
from datasets import load_dataset


def collate_fn(batch):
    # 배치의 각 요소를 텐서로 변환
    input_ids = torch.stack([torch.tensor(item['input_ids']) for item in batch])
    attention_mask = torch.stack([torch.tensor(item['attention_mask']) for item in batch])
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }

def pretrain():
    # GPU 설정
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('mps:0')

    # BERT Base 토크나이저 및 모델 로드
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForMaskedLM.from_pretrained('bert-base-uncased').to(device)
    model.init_weights()

    # 데이터셋 로드 (IMDb 리뷰 데이터셋, 5000건 이하)
    dataset = load_dataset('stanfordnlp/imdb', split='train').shuffle(seed=42).select(range(5000))

    # 데이터 전처리 함수
    def preprocess_function(examples):
        return tokenizer(
            examples['text'], 
            truncation=True, 
            padding='max_length', 
            max_length=512
        )

    # 데이터셋 토큰화
    tokenized_dataset = dataset.map(preprocess_function, batched=True)

    # DataLoader 생성
    dataloader = DataLoader(
        tokenized_dataset, 
        batch_size=16, 
        shuffle=True,
        collate_fn=collate_fn  # 이 부분 추가
    )

    # 옵티마이저 설정
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # 사전학습 루프
    num_epochs = 1
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch in dataloader:
            # 배치 데이터 GPU로 이동
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # 모델 forward pass
            outputs = model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                labels=input_ids
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            # 역전파 및 가중치 업데이트
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # 에폭별 평균 손실 출력
        print(f'Epoch {epoch+1}, Average Loss: {total_loss/len(dataloader)}')

    # 모델 저장
    model.save_pretrained('./models/bert_pretrained_model')
    tokenizer.save_pretrained('./models/bert_pretrained_model')


if __name__ == "__main__":
    pretrain()