import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import joblib

# Load intents
with open('intents.json') as file:
    intents = json.load(file)

# Extract patterns and intents
patterns = []
intent_labels = []
for intent in intents['intents']:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        intent_labels.append(intent['tag'])

print(f"Total patterns: {len(patterns)}, Total intents: {len(set(intent_labels))}")

# Vectorize patterns
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(patterns).toarray()

# Encode intents
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(intent_labels)

# Split data (reduce test size for small dataset)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

# Train model with simpler architecture
mlp = MLPClassifier(hidden_layer_sizes=(32,), max_iter=1000, activation='relu', solver='adam', random_state=42)
mlp.fit(X_train, y_train)

# Check accuracy
y_pred = mlp.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Save everything
joblib.dump(mlp, 'chatbot_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

print("Files saved: chatbot_model.pkl, vectorizer.pkl, label_encoder.pkl")