from transformers import ModernBertForMaskedLM

model = ModernBertForMaskedLM.from_pretrained("answerdotai/ModernBERT-base")
# model = ModernBertForMaskedLM.from_pretrained("google-bert/bert-base-uncased")
# model = ModernBertForMaskedLM.from_pretrained("./models/modernbert-base-en")

for x in list(model.state_dict().keys()):
    print(x)
