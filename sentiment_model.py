# sentiment_model.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# Load multilingual sentiment model
MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

labels = ['Negative', 'Neutral', 'Positive']

def analyze_sentiment_ml(review_text):
    """
    Input: review_text (English + Telugu mix)
    Output: 'Positive' / 'Negative' / 'Neutral'
    """
    inputs = tokenizer(review_text, return_tensors="pt", truncation=True, max_length=128)
    outputs = model(**inputs)
    scores = F.softmax(outputs.logits, dim=1)
    sentiment = labels[scores.argmax()]
    return sentiment
