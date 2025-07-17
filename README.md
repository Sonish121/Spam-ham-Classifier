# Email Spam/Ham Classifier

This project is a machine learning pipeline to classify emails (or SMS messages) as either **Spam** (unsolicited, malicious, or advertising) or **Ham** (legitimate, non-spam). It demonstrates data preprocessing, feature extraction, model training, evaluation, interactive prediction using multiple algorithms, and both command-line and web interfaces.

## Features
- Uses the [UCI SMS Spam Collection Dataset](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)
- Text preprocessing (lowercasing, punctuation removal, stopword removal)
- Feature extraction with TF-IDF vectorization using n-grams (unigrams and bigrams)
- Handles class imbalance with `class_weight='balanced'` for Logistic Regression and Random Forest
- Trains and compares three models:
  - Multinomial Naive Bayes
  - Logistic Regression
  - Random Forest
- Evaluates models using accuracy, precision, recall, and F1-score
- Threshold tuning for spam detection sensitivity (default threshold: 0.15 for higher spam sensitivity)
- Command-line interface (CLI) for interactive predictions
- Web interface using Flask for easy browser-based predictions
- Saves trained models and vectorizer for future use

## Setup
1. **Clone the repository** and navigate to the project directory.
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Download the dataset:**
   - Download the [SMSSpamCollection](https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip) file, extract it, and place the `SMSSpamCollection` file in the project directory.

## Usage
### 1. Train and Evaluate Models
Run the main script to preprocess data, train models, evaluate, and save them:
```bash
python spam_classifier.py
```
- The script will:
  - Preprocess the data
  - Train and evaluate all three models
  - Print detailed metrics for each model
  - Save the trained models and vectorizer as `.joblib` files
  - Allow you to interactively classify new messages via the CLI

### 2. Command-Line Interface (CLI)
After training, the script enters CLI mode:
```
--- Spam/Ham Classifier CLI ---
Type your message and press Enter to classify it as spam or ham.
Type 'exit' to quit.

Enter a message: Congratulations! You've won a free ticket. Call now!
Prediction (Naive Bayes, threshold=0.15): spam (spam probability: 0.22)
Prediction (Logistic Regression, threshold=0.15): spam (spam probability: 0.41)
Prediction (Random Forest, threshold=0.15): spam (spam probability: 0.19)
----------------------------------------
```

### 3. Web Interface (Flask)
You can use a web interface for easy predictions:
1. Make sure you have run `python spam_classifier.py` to save the models.
2. Start the Flask app:
   ```bash
   python app.py
   ```
3. Open [http://127.0.0.1:5000/](http://127.0.0.1:5000/) in your browser.
4. Enter a message and see predictions from all three models.

### 4. Saving and Loading Models
- The script saves the trained models and vectorizer as `.joblib` files for future use.
- To load and use them in another script:
  ```python
  import joblib
  clf_nb = joblib.load('model_naive_bayes.joblib')
  clf_lr = joblib.load('model_logistic_regression.joblib')
  clf_rf = joblib.load('model_random_forest.joblib')
  vectorizer = joblib.load('vectorizer.joblib')
  ```

## Model Improvements
- **N-gram Features:** The model uses both unigrams and bigrams for richer text representation.
- **Class Imbalance Handling:** Logistic Regression and Random Forest use `class_weight='balanced'` to better detect spam in imbalanced data.
- **Threshold Tuning:** The spam threshold is set to 0.15 by default, making the model more sensitive to spam.
- **Web Interface:** A Flask app provides a user-friendly way to classify messages in your browser.

## Model Evaluation
- The script prints accuracy, precision, recall, and F1-score for the spam class for each model.
- Example output:
```
Naive Bayes Metrics (spam class):
Accuracy: 0.9704
Precision: 1.0000
Recall: 0.7800
F1-score: 0.8764
```

## Customization
- You can adjust the spam threshold in both CLI and web app for more or less aggressive spam detection.
- You can easily extend the script to use other models or add more advanced features.

## Project Workflow
1. **Preprocess** the dataset (cleaning, stopword removal)
2. **Extract features** using TF-IDF with n-grams
3. **Train** three models (Naive Bayes, Logistic Regression, Random Forest)
4. **Evaluate** models with multiple metrics
5. **Save** models and vectorizer
6. **Predict** new messages via CLI or web interface

## Deployment (Host Online on Render)

This project is hosted online using [Render](https://spam-ham-classifier-chmg.onrender.com/). 


## License
This project is for educational purposes. 