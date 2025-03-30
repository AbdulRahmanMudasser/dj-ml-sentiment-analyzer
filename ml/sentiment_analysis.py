import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Sample Data
df = pd.DataFrame({
    'text': ['I love this', 'Terrible experience', 'Awesome service', 'Waste of time'],
    'sentiment': ['positive', 'negative', 'positive', 'negative']
})

# Preprocessing
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['text'])
Y = df['sentiment']

# Train Model
model = LogisticRegression()
model.fit(X, Y)

# Save Model
joblib.dump(model, 'trained models/sentiment_model.joblib')
joblib.dump(vectorizer, 'trained models/tfidf_vectorizer.joblib')