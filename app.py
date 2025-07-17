from flask import Flask, render_template, request
import joblib
import os
import string
import nltk
from nltk.corpus import stopwords

# Ensure NLTK stopwords are downloaded
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

# Load models and vectorizer
clf_nb = joblib.load('model_naive_bayes.joblib')
clf_lr = joblib.load('model_logistic_regression.joblib')
clf_rf = joblib.load('model_random_forest.joblib')
vectorizer = joblib.load('vectorizer.joblib')

# Preprocessing function (should match your training script)
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

def predict_message(message, model='nb', spam_threshold=0.15):
    clean = preprocess_text(message)
    vect = vectorizer.transform([clean])
    if model == 'lr':
        proba = clf_lr.predict_proba(vect)[0]
        pred = 'spam' if proba[1] > spam_threshold else 'ham'
        prob = proba[1]
        model_name = 'Logistic Regression'
    elif model == 'rf':
        proba = clf_rf.predict_proba(vect)[0]
        pred = 'spam' if proba[1] > spam_threshold else 'ham'
        prob = proba[1]
        model_name = 'Random Forest'
    else:
        proba = clf_nb.predict_proba(vect)[0]
        pred = 'spam' if proba[1] > spam_threshold else 'ham'
        prob = proba[1]
        model_name = 'Naive Bayes'
    return model_name, pred, prob

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    message = ''
    if request.method == 'POST':
        message = request.form['message']
        results = []
        for model in ['nb', 'lr', 'rf']:
            model_name, pred, prob = predict_message(message, model=model, spam_threshold=0.15)
            results.append({'model': model_name, 'prediction': pred, 'probability': f'{prob:.2f}'})
        result = results
    return render_template('index.html', result=result, message=message)

if __name__ == '__main__':
    app.run(debug=True)