import os
from django.conf import settings
from django.shortcuts import render
import joblib

BASE_DIR = os.path.dirname(settings.BASE_DIR)

model_path = os.path.join(BASE_DIR, 'ml', 'trained models', 'sentiment_model.joblib')
vectorizer_path = os.path.join(BASE_DIR, 'ml', 'trained models', 'tfidf_vectorizer.joblib')

# Sentiment Analyzer Function
def sentiment_analyzer(text):
    # Load Models
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    
    processed_text = vectorizer.transform([text])
    
    return model.predict(processed_text)[0]
    
def home(request):
    result = None
    
    if request.method == "POST":
        text = request.POST.get('text')
        result = sentiment_analyzer(text)
    
    return render(request, 'analyzer/home.html', {'result': result})
