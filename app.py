import os
import nltk
import gdown
from flask import Flask, request, render_template, jsonify
import joblib
import re
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

# Download NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('stopwords', quiet=True)
except Exception as e:
    print(f"Error downloading NLTK data: {e}")

# Download .pkl files if missing (replace with your Google Drive links)
files = [
    ('YOUR_GOOGLE_DRIVE_LINK1', 'fake_news_model.pkl'),
    ('YOUR_GOOGLE_DRIVE_LINK2', 'tfidf_vectorizer.pkl'),
    ('YOUR_GOOGLE_DRIVE_LINK3', 'chatbot_model.pkl'),
    ('YOUR_GOOGLE_DRIVE_LINK4', 'vectorizer.pkl'),
    ('YOUR_GOOGLE_DRIVE_LINK5', 'label_encoder.pkl')
]
for url, file in files:
    if not os.path.exists(file):
        try:
            gdown.download(url, file, quiet=False)
        except Exception as e:
            print(f"Error downloading {file}: {e}")

# Load models
try:
    model = joblib.load('fake_news_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    chatbot_model = joblib.load('chatbot_model.pkl')  # If needed by chatbot.py
    chatbot_vectorizer = joblib.load('vectorizer.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
except FileNotFoundError as e:
    print(f"Error: Model or vectorizer file not found: {e}")
    exit()

app = Flask(__name__, static_folder='static', template_folder='templates')

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
        response = requests.get(url, timeout=10)
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
try:
    explainer_shap = shap.TreeExplainer(model, background_features)
except Exception as e:
    print(f"Error initializing SHAP explainer: {e}")

# Twitter API credentials (replace with environment variables on Render)
consumer_key = os.environ.get('TWITTER_CONSUMER_KEY', 'zMQRY3egXLKXAvWh1lY54FstZ')
consumer_secret = os.environ.get('TWITTER_CONSUMER_SECRET', '1xHqbl5ljlo50G7z4XjunTVfhmHzTUFq4L38glGI1yhgr5G537')
access_token = os.environ.get('TWITTER_ACCESS_TOKEN', '1624944816139149314-T1otmTBecm3ISNjbAmcZcgNFwyaWaM')
access_token_secret = os.environ.get('TWITTER_ACCESS_TOKEN_SECRET', 'jF2RT0qP8mLzdsEioeD4cHJUwoG9ae1ZKL1brUx6PymwB')

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
                try:
                    exp = explainer.explain_instance(text, classifier_fn, num_features=5)
                    explanation = exp.as_list()
                except Exception as e:
                    explanation = [(f"Error generating LIME explanation: {e}", 0)]
                shap_values = explainer_shap.shap_values(features) if 'explainer_shap' in globals() else []
                shap_summary = []
                if hasattr(shap_values, 'shape') and len(shap_values.shape) == 2:
                    for i in range(len(vectorizer.get_feature_names_out())):
                        try:
                            shap_value = float(shap_values[0][i])
                            if abs(shap_value) > 0.01:
                                shap_summary.append((vectorizer.get_feature_names_out()[i], shap_value))
                        except IndexError:
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
                    try:
                        exp = explainer.explain_instance(text, classifier_fn, num_features=5)
                        explanation = exp.as_list()
                    except Exception as e:
                        explanation = [(f"Error generating LIME explanation: {e}", 0)]
                    shap_values = explainer_shap.shap_values(features) if 'explainer_shap' in globals() else []
                    shap_summary = []
                    if hasattr(shap_values, 'shape') and len(shap_values.shape) == 2:
                        for i in range(len(vectorizer.get_feature_names_out())):
                            try:
                                shap_value = float(shap_values[0][i])
                                if abs(shap_value) > 0.01:
                                    shap_summary.append((vectorizer.get_feature_names_out()[i], shap_value))
                            except IndexError:
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
                            try:
                                exp = explainer.explain_instance(text.strip(), classifier_fn, num_features=5)
                                explanation = exp.as_list()
                            except Exception as e:
                                explanation = [(f"Error generating LIME explanation: {e}", 0)]
                            shap_values = explainer_shap.shap_values(features) if 'explainer_shap' in globals() else []
                            shap_summary = []
                            if hasattr(shap_values, 'shape') and len(shap_values.shape) == 2:
                                for i in range(len(vectorizer.get_feature_names_out())):
                                    try:
                                        shap_value = float(shap_values[0][i])
                                        if abs(shap_value) > 0.01:
                                            shap_summary.append((vectorizer.get_feature_names_out()[i], shap_value))
                                    except IndexError:
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
        try:
            exp = explainer.explain_instance(text, classifier_fn, num_features=5)
            explanation = exp.as_list()
        except Exception as e:
            explanation = [(f"Error generating LIME explanation: {e}", 0)]
        shap_values = explainer_shap.shap_values(features) if 'explainer_shap' in globals() else []
        shap_summary = []
        for i in range(len(vectorizer.get_feature_names_out())):
            try:
                shap_value = float(shap_values[0][i])
                if abs(shap_value) > 0.01:
                    shap_summary.append((vectorizer.get_feature_names_out()[i], shap_value))
            except IndexError:
                continue
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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))