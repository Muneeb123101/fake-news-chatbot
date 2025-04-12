from flask import Flask, request, render_template, jsonify
import joblib
import re
import nltk
import os
from werkzeug.utils import secure_filename
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import requests
from bs4 import BeautifulSoup
from lime.lime_text import LimeTextExplainer
import shap
import tweepy
from tweepy import errors
from chatbot import predict_class, get_response  # Import custom chatbot

app = Flask(__name__)

# Load the saved model and vectorizer
try:
    model = joblib.load('fake_news_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
except FileNotFoundError:
    print("Error: Model or vectorizer file not found. Run the training script first.")
    exit()

# Set up cleaning tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Function to clean text
def preprocess_text(text):
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d+', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Function to scrape text from a URL
def scrape_text_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        text = soup.get_text(separator=' ')
        text = ' '.join(text.split())
        return text
    except requests.RequestException as e:
        print(f"Error fetching URL: {e}")
        return None

# Define background texts for SHAP
background_texts = [
    "This is a real news article about politics.",
    "This is a fake news story designed to mislead.",
    "Another example of real news.",
    "A fabricated story with no basis in reality."
]

# Preprocess and vectorize background texts
background_clean = [preprocess_text(text) for text in background_texts]
background_features = vectorizer.transform(background_clean).toarray()

# Initialize LIME and SHAP explainers
explainer = LimeTextExplainer(class_names=['Real', 'Fake'])
explainer_shap = shap.TreeExplainer(model, background_features)

# Twitter API credentials
consumer_key = 'zMQRY3egXLKXAvWh1lY54FstZ'
consumer_secret = '1xHqbl5ljlo50G7z4XjunTVfhmHzTUFq4L38glGI1yhgr5G537'
access_token = '1624944816139149314-T1otmTBecm3ISNjbAmcZcgNFwyaWaM'
access_token_secret = 'jF2RT0qP8mLzdsEioeD4cHJUwoG9ae1ZKL1brUx6PymwB'

# Authenticate with Twitter API
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

# Function to get tweet text from a URL
def get_tweet_text(tweet_url):
    try:
        tweet_id = tweet_url.split('/')[-1]
        tweet = api.get_status(tweet_id)
        return tweet.text
    except tweepy.errors.TweepyException as e:
        print(f"Error fetching tweet: {e}")
        return None

# Classifier function for LIME
def classifier_fn(texts):
    clean_texts = [preprocess_text(text) for text in texts]
    features = vectorizer.transform(clean_texts).toarray()
    return model.predict_proba(features)

# Ensure the uploads folder exists
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

ALLOWED_EXTENSIONS = {'txt'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    single_result = None
    batch_results = None
    error = None
    labels = []
    confidences = []
    chat_history = request.args.get('chat_history', '')

    if request.method == 'POST':
        if 'submit_text' in request.form:
            text = request.form['text']
            if text.strip():
                clean_text = preprocess_text(text)
                features = vectorizer.transform([clean_text]).toarray()
                prediction = model.predict(features)[0]
                probability = model.predict_proba(features)[0][prediction] * 100
                exp = explainer.explain_instance(text, classifier_fn, num_features=5)
                explanation = exp.as_list()
                shap_values = explainer_shap.shap_values(features)
                shap_summary = []
                if hasattr(shap_values, 'shape') and len(shap_values.shape) == 2:
                    for i in range(len(vectorizer.get_feature_names_out())):
                        try:
                            shap_value = float(shap_values[0][i])
                            if abs(shap_value) > 0.01:
                                shap_summary.append((vectorizer.get_feature_names_out()[i], shap_value))
                        except IndexError:
                            print(f"IndexError: Could not access index {i} in shap_values")
                            continue
                shap_summary = shap_summary[:5]
                single_result = {
                    'prediction': 'Fake' if prediction == 1 else 'Real', 
                    'confidence': round(probability, 2),
                    'explanation': explanation,
                    'shap_summary': shap_summary
                }
            else:
                error = "Please enter some text to analyze."

        elif 'submit_url' in request.form:
            url = request.form['url']
            if url.strip():
                text = scrape_text_from_url(url)
                if text:
                    clean_text = preprocess_text(text)
                    features = vectorizer.transform([clean_text]).toarray()
                    prediction = model.predict(features)[0]
                    probability = model.predict_proba(features)[0][prediction] * 100
                    exp = explainer.explain_instance(text, classifier_fn, num_features=5)
                    explanation = exp.as_list()
                    shap_values = explainer_shap.shap_values(features)
                    shap_summary = []
                    if hasattr(shap_values, 'shape') and len(shap_values.shape) == 2:
                        for i in range(len(vectorizer.get_feature_names_out())):
                            try:
                                shap_value = float(shap_values[0][i])
                                if abs(shap_value) > 0.01:
                                    shap_summary.append((vectorizer.get_feature_names_out()[i], shap_value))
                            except IndexError:
                                print(f"IndexError: Could not access index {i} in shap_values")
                                continue
                    shap_summary = shap_summary[:5]
                    single_result = {
                        'prediction': 'Fake' if prediction == 1 else 'Real', 
                        'confidence': round(probability, 2),
                        'explanation': explanation,
                        'shap_summary': shap_summary
                    }
                else:
                    error = "Could not retrieve text from the URL."
            else:
                error = "Please enter a URL to analyze."

        elif 'submit_file' in request.form:
            file = request.files.get('file')
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(UPLOAD_FOLDER, filename)
                file.save(file_path)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        texts = f.readlines()
                    batch_results = []
                    for text in texts:
                        clean_text = preprocess_text(text.strip())
                        if clean_text:
                            features = vectorizer.transform([clean_text]).toarray()
                            prediction = model.predict(features)[0]
                            probability = model.predict_proba(features)[0][prediction] * 100
                            exp = explainer.explain_instance(text.strip(), classifier_fn, num_features=5)
                            explanation = exp.as_list()
                            shap_values = explainer_shap.shap_values(features)
                            shap_summary = []
                            if hasattr(shap_values, 'shape') and len(shap_values.shape) == 2:
                                for i in range(len(vectorizer.get_feature_names_out())):
                                    try:
                                        shap_value = float(shap_values[0][i])
                                        if abs(shap_value) > 0.01:
                                            shap_summary.append((vectorizer.get_feature_names_out()[i], shap_value))
                                    except IndexError:
                                        print(f"IndexError: Could not access index {i} in shap_values")
                                        continue
                            shap_summary = shap_summary[:5]
                            batch_results.append({
                                'text': text.strip(), 
                                'prediction': 'Fake' if prediction == 1 else 'Real', 
                                'confidence': round(probability, 2),
                                'explanation': explanation,
                                'shap_summary': shap_summary
                            })
                    labels = [f"Article {i+1}" for i in range(len(batch_results))]
                    confidences = [res['confidence'] for res in batch_results]
                except UnicodeDecodeError:
                    error = "The file contains invalid characters. Please upload a valid UTF-8 encoded file."
                except Exception as e:
                    error = f"An error occurred while processing the file: {e}"
            else:
                error = "Invalid file. Please upload a .txt file."

    return render_template('index.html', single_result=single_result, batch_results=batch_results, 
                          labels=labels, confidences=confidences, error=error, chat_history=chat_history)

@app.route('/predict_tweet', methods=['POST'])
def predict_tweet():
    tweet_url = request.form['tweet_url']
    text = get_tweet_text(tweet_url)
    if text:
        clean_text = preprocess_text(text)
        features = vectorizer.transform([clean_text]).toarray()
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][prediction] * 100
        exp = explainer.explain_instance(text, classifier_fn, num_features=5)
        explanation = exp.as_list()
        shap_values = explainer_shap.shap_values(features)
        shap_summary = []
        for i in range(len(vectorizer.get_feature_names_out())):
            shap_value = float(shap_values[0][i])
            if abs(shap_value) > 0.01:
                shap_summary.append((vectorizer.get_feature_names_out()[i], shap_value))
        shap_summary = shap_summary[:5]
        single_result = {
            'prediction': 'Fake' if prediction == 1 else 'Real',
            'confidence': round(probability, 2),
            'explanation': explanation,
            'shap_summary': shap_summary
        }
        return render_template('index.html', single_result=single_result)
    return render_template('index.html', error="Error fetching tweet.")

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.form.get('message', '')
    chat_history = request.form.get('chat_history', '')
    
    if user_message.strip():
        # Use custom chatbot
        intent = predict_class(user_message)
        response = get_response(intent)
        chat_history += f"User: {user_message}\nBot: {response}\n"
    else:
        response = "Please type a message!"
        chat_history += f"User: {user_message}\nBot: {response}\n"
    
    return jsonify({'response': response, 'chat_history': chat_history})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))