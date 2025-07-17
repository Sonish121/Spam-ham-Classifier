import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib


# Define a preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# Load the dataset
df = pd.read_csv('SMSSpamCollection', sep='\t', header=None, names=['label', 'text'])

# Show the first 5 rows
# print(df.head())
# print(f"\nDataset shape: {df.shape}")
# print(f"Label distribution:\n{df['label'].value_counts()}")

# Apply preprocessing
df['clean_text'] = df['text'].apply(preprocess_text)

# Show the first 5 preprocessed messages
# print("\nPreprocessed messages:")
# print(df[['text', 'clean_text']].head())

# 4.1 Convert text to features
vectorizer = TfidfVectorizer(ngram_range=(1,2))
X = vectorizer.fit_transform(df['clean_text'])
y = df['label']

# 4.2 Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4.3 Train the classifiers
clf_nb = MultinomialNB()
clf_nb.fit(X_train, y_train)

clf_lr = LogisticRegression(max_iter=1000, class_weight='balanced')
clf_lr.fit(X_train, y_train)

clf_rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
clf_rf.fit(X_train, y_train)

def print_metrics(y_true, y_pred, model_name):
    print(f"\n{model_name} Metrics (spam class):")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, pos_label='spam', average='binary'):.4f}")
    print(f"Recall: {recall_score(y_true, y_pred, pos_label='spam', average='binary'):.4f}")
    print(f"F1-score: {f1_score(y_true, y_pred, pos_label='spam', average='binary'):.4f}")

# Predict on test set
print("\n--- Naive Bayes Results ---")
y_pred_nb = clf_nb.predict(X_test)
print(classification_report(y_test, y_pred_nb))
print(f"Accuracy: {accuracy_score(y_test, y_pred_nb):.4f}")
print_metrics(y_test, y_pred_nb, "Naive Bayes")

print("\n--- Logistic Regression Results ---")
y_pred_lr = clf_lr.predict(X_test)
print(classification_report(y_test, y_pred_lr))
print(f"Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")
print_metrics(y_test, y_pred_lr, "Logistic Regression")

print("\n--- Random Forest Results ---")
y_pred_rf = clf_rf.predict(X_test)
print(classification_report(y_test, y_pred_rf))
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print_metrics(y_test, y_pred_rf, "Random Forest")

# Save models and vectorizer
joblib.dump(clf_nb, 'model_naive_bayes.joblib')
joblib.dump(clf_lr, 'model_logistic_regression.joblib')
joblib.dump(clf_rf, 'model_random_forest.joblib')
joblib.dump(vectorizer, 'vectorizer.joblib')
print("Models and vectorizer saved!")

# Updated prediction function to allow model selection
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
    print(f"\nMessage: {message}")
    print(f"Prediction ({model_name}, threshold={spam_threshold}): {pred} (spam probability: {prob:.2f})")

# Example usage:
if __name__ == "__main__":
    predict_message("Congratulations! You've won a free ticket. Call now!", model='nb', spam_threshold=0.3)
    predict_message("Congratulations! You've won a free ticket. Call now!", model='lr', spam_threshold=0.3)
    predict_message("Congratulations! You've won a free ticket. Call now!", model='rf', spam_threshold=0.3)
    predict_message("Hey, are we still meeting for lunch today?", model='nb', spam_threshold=0.3)
    predict_message("Hey, are we still meeting for lunch today?", model='lr', spam_threshold=0.3)
    predict_message("Hey, are we still meeting for lunch today?", model='rf', spam_threshold=0.3)

    print("\n--- Spam/Ham Classifier CLI ---")
    print("Type your message and press Enter to classify it as spam or ham.")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("Enter a message: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        predict_message(user_input, model='nb', spam_threshold=0.3)
        predict_message(user_input, model='lr', spam_threshold=0.3)
        predict_message(user_input, model='rf', spam_threshold=0.3)
        print("-" * 40)