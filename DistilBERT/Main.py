import torch
import pandas as pd
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# Prepare Dataset for Training and Testing
class ReviewDataset(Dataset):
    def __init__(self, reviews, sentiments):
        self.reviews = reviews
        self.sentiments = sentiments
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        review = self.reviews[idx]
        sentiment = self.sentiments[idx]
        encoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=512,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            truncation=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(sentiment, dtype=torch.long)
        }

# Accuracy and F1 Metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1
    }

def main():
    # Load the dataset
    dataset_path = 'Data/IMDB Dataset - 10000.csv'
    data = pd.read_csv(dataset_path)
    data['sentiment'] = data['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)

    # Set up dataset split
    train_data, eval_data = train_test_split(data, test_size=0.3, random_state=42)

    # Assign data to train/test splits
    train_dataset = ReviewDataset(train_data['review'].tolist(), train_data['sentiment'].tolist())
    eval_dataset = ReviewDataset(eval_data['review'].tolist(), eval_data['sentiment'].tolist())

    # Prefer CUDA GPU training over CPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model and send to GPU/CPU
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
    model.to(device)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        # Load only the best model at the end of all training epochs
        load_best_model_at_end=True,
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics
    )

    # Train
    trainer.train()

    # Evaluate
    results = trainer.evaluate()
    print(results)

if __name__ == '__main__':
    main()