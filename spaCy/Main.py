import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import re

# Result display formatting
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 100)

# Prefer GPU training over CPU if available
activated = spacy.prefer_gpu()

# Load spaCy English transformer model
nlp = spacy.load('en_core_web_trf')

# Append SpacyTextBlob to spaCy pipeline for sentiment analysis
spacy_text_blob = SpacyTextBlob(nlp)
nlp.add_pipe('spacytextblob')


# Clean input data
def clean_text(text):
    # remove html line breaks
    text = re.sub(r'<br\s*/?>', '', text)

    #remove non-alphanumeric characters
    text = re.sub(r"[^a-zA-Z0-9]", ' ', text)

    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()

    # Lemmatization using SpaCy tokens
    cleaning_doc = nlp(text)
    lemmatized_text = ' '.join([token.lemma_ for token in cleaning_doc])

    # return cleaned and lemmatized text
    return lemmatized_text



# Function to analyze sentiment using SpaCyTextBlob
def analyze_sentiment(text):
    # Tokenize and process input
    doc = nlp(text)

    # Return the sentiment polarity
    return doc._.blob.sentiment.polarity


# Load data
def load_data(filepath):
    # Using pandas to load dataset
    data = pd.read_csv(filepath)
    data['review'] = data['review'].apply(clean_text)

    # Return pandas dataframe
    return data


def perform_sentiment_analysis(filepath):
    data = load_data(filepath)
    # 7-fold cross-validation
    kf = KFold(n_splits=7)
    accuracies = []
    f1_scores = []
    # List to store predictions
    all_predictions = []

    # Prepare data
    for train_index, test_index in kf.split(data):
        test_reviews = data.iloc[test_index]['review']
        y_test = data.iloc[test_index]['sentiment'].apply(lambda x: 1 if x == 'positive' else 0).values
        test_sentiments = [analyze_sentiment(review) for review in test_reviews]
        predictions = ['positive' if score > 0 else 'negative' for score in test_sentiments]
        predicted_labels = [1 if pred == 'positive' else 0 for pred in predictions]

        # Store predictions
        for review, actual, predicted, rating in zip(test_reviews, y_test, predicted_labels, test_sentiments):
            all_predictions.append({'Review': review, 'Actual': actual, 'Predicted': predicted, 'Rating': ((rating+1)/2)})

        # Calculate metrics
        acc = accuracy_score(y_test, predicted_labels)
        f1 = f1_score(y_test, predicted_labels)
        accuracies.append(acc)
        f1_scores.append(f1)


    # Convert predictions list to DataFrame, save, and display results
    predictions_df = pd.DataFrame(all_predictions)
    predictions_df.to_csv('movie_review_predictions.csv', index=False)
    print(predictions_df.head(15))  # Optionally print the first few rows

    # Print results
    print(f"Average Model Accuracy: {np.mean(accuracies):.4f}")
    print(f"Average Model F1 Score: {np.mean(f1_scores):.4f}")

# Model training
if __name__ == "__main__":
    perform_sentiment_analysis('Data/IMDB Dataset - 10000.csv')