from transformers import AutoModel, AutoTokenizer

# tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
tokenizer = AutoTokenizer.from_pretrained("./models/modernbert-base-kr")
tokens = tokenizer.encode("Hi.")
print(tokens)
