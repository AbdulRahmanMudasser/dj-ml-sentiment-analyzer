import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Sample Data
df = pd.DataFrame({
    'text': [
        'I love this product!', 'Excellent service', 'Wonderful experience',
        'Highly recommended', 'Terrible quality', 'Worst purchase ever',
        'Disappointing', 'Absolutely hated it', '', ' ',  
    ],
    'sentiment': [
        'positive', 'positive', 'positive', 'positive',
        'negative', 'negative', 'negative', 'negative',
        'neutral', 'neutral'
    ]
})

# Handle empty strings explicitly
df['text'] = df['text'].str.strip().replace(r'^\s*$', 'neutral', regex=True)

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