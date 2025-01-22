from transformers import BertTokenizer, BertForQuestionAnswering
from transformers import pipeline

def qtest():
    # 모델과 토크나이저 로드
    model_name = "google-bert/bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForQuestionAnswering.from_pretrained(model_name)

    # QA 파이프라인 생성
    qa_pipeline = pipeline(
        "question-answering",
        model=model,
        tokenizer=tokenizer
    )

    # 예시 컨텍스트와 질문
    context = "Stanford Question Answering Dataset (SQuAD)는 크라우드워커들이 위키피디아 문서에 대해 작성한 질문-답변 쌍으로 구성된 독해 데이터셋입니다."
    question = "SQuAD는 어떤 데이터셋인가요?"

    # 예측 수행
    result = qa_pipeline({
        'question': question,
        'context': context
    })

    print(f"답변: {result['answer']}")
    print(f"점수: {result['score']:.2f}")


if __name__ == '__main__':
    qtest()