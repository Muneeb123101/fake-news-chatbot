import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Load the dataset
try:
    data = pd.read_csv('train.csv')
except FileNotFoundError:
    print("Error: 'train.csv' not found. Please ensure the file is in the same directory.")
    exit()

# Set up cleaning tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Function to clean text
def preprocess_text(text):
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = re.sub(r'\d+', '', text)  # Remove numbers
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Clean the 'text' column
if 'text' not in data.columns or 'label' not in data.columns:
    print("Error: 'train.csv' must contain 'text' and 'label' columns.")
    exit()
data['clean_text'] = data['text'].apply(preprocess_text)
data = data.dropna(subset=['text'])  # Remove rows with missing text

# Turn text into numbers
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(data['clean_text']).toarray()
y = data['label']

# Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model and vectorizer
joblib.dump(model, 'fake_news_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

# Function to predict new text
def predict_fake_news(text):
    clean_text = preprocess_text(text)
    features = vectorizer.transform([clean_text]).toarray()
    prediction = model.predict(features)[0]
    return 'Fake' if prediction == 1 else 'Real'

# Test with a sample
sample_text = "Breaking: Aliens land on White House lawn!"
print("Prediction:", predict_fake_news(sample_text))