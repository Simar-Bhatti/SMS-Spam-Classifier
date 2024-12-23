# app.py

from flask import Flask, request, render_template
import pickle
import numpy as np

# Load the trained model and vectorizer
model_path = 'model.pkl'
vectorizer_path = 'vectorizer.pkl'

with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

with open(vectorizer_path, 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract the text input from the form
    user_input = request.form['text']
    print(f"User input: {user_input}")  # Log the input

    # Convert the text input into a format the model can understand using the TF-IDF vectorizer
    text_vectorized = vectorizer.transform([user_input])
    print(f"Vectorized input: {text_vectorized.toarray()}")  # Log the vectorized input

    # Make prediction using the trained model
    prediction = model.predict(text_vectorized)
    print(f"Prediction result: {prediction}")  # Log the prediction

    # Ensure the prediction result is properly interpreted
    if prediction[0] == 1:
        prediction_text = 'Spam'
    else:
        prediction_text = 'Not Spam'

    print(f"Prediction output: {prediction_text}")  # Log the final output

    return render_template('index.html', prediction_text=f'Prediction: {prediction_text}')

if __name__ == "__main__":
    app.run(debug=True)
