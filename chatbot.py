import json
import random
import numpy as np
import joblib

# Load intents
with open('intents.json') as file:
    intents = json.load(file)

# Load trained model and tools
mlp = joblib.load('chatbot_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Predict intent
def predict_class(sentence, threshold=0.7):
    X_input = vectorizer.transform([sentence]).toarray()
    probs = mlp.predict_proba(X_input)[0]
    max_prob = np.max(probs)
    if max_prob > threshold:
        pred = np.argmax(probs)
        return label_encoder.inverse_transform([pred])[0]
    return "unknown"

# Get response
def get_response(intent_tag):
    for intent in intents['intents']:
        if intent['tag'] == intent_tag:
            return random.choice(intent['responses'])
    return "Sorry, I didn’t understand that. Ask me about fake news!"

# Test
while True:
    message = input("You: ")
    if message.lower() == "quit":
        break
    print("Bot:", get_response(predict_class(message)))