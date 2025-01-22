import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("mps:0")

model_id = "clapAI/modernBERT-base-multilingual-sentiment"
# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id, torch_dtype=torch.float16)

model.to(device)
model.eval()


# Retrieve labels from the model's configuration
id2label = model.config.id2label

texts = [
    # English
    {
        "text": "I absolutely love the new design of this app!",
        "label": "positive"
    },
    {
        "text": "The customer service was disappointing.",
        "label": "negative"
    },
    # Korean
    {
        "text": "나는 정말로 당신을 사랑하지 않아요 !",
        "label": "positive"
    },
    {
        "text": "고객 서비스가 정말 실망스러웠어요.",
        "label": "negative"
    },

]

for item in texts:
    text = item["text"]
    label = item["label"]

    inputs = tokenizer(text, return_tensors="pt").to(device)

    # Perform inference in inference mode
    with torch.inference_mode():
        outputs = model(**inputs)
        predictions = outputs.logits.argmax(dim=-1)
    print(f"Text: {text} | Label: {label} | Prediction: {id2label[predictions.item()]}")
