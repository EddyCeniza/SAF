import torch

from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

# Path to saved pre-trained model
model_path = './10k reviews results/checkpoint-1500'
model = DistilBertForSequenceClassification.from_pretrained(model_path)

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')


def predict(texts):
    # Tokenize input
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=512, return_tensors='pt')

    # Get predictions
    # Deactivates autograd, help with training speed and memory usage
    with torch.no_grad():
        outputs = model(**encodings)

    # Process outputs
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=-1)

    # Convert to list and return
    return probabilities.tolist()


# Input test text and run model
test_input = ["This movie was okay. I probably would not watch it again."]
predictions = predict(test_input)

print(predictions)
