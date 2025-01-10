# model="google-bert/bert-base-multilingual-uncased",

from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('google-bert/bert-base-multilingual-uncased')
model = BertModel.from_pretrained("google-bert/bert-base-multilingual-uncased")
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')
print(encoded_input)