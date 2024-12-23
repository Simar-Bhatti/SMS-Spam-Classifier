This repository contains a machine learning model designed to classify SMS messages as either spam or ham (non-spam). The model is trained using the Multinomial Naive Bayes algorithm and can predict the nature of a message based on its text content. It is deployed on a web interface for easy and instant use.

Key Features:
Text Preprocessing: The model performs essential data cleaning steps including tokenization, converting text to lowercase, and removing stop words to ensure effective feature extraction.
Feature Extraction: The TF-IDF (Term Frequency-Inverse Document Frequency) technique is used to convert the SMS text into numerical features, capturing the significance of each word in the dataset.
Model Training: Trained using the Multinomial Naive Bayes classifier, the model learns to differentiate between spam and ham messages based on the content of the SMS.
Model Deployment: The model is deployed on a simple web interface, allowing users to input SMS messages and instantly receive predictions on whether the message is spam or ham.
Try It Out!

The dataset used for training this model is sourced from Kaggle. It contains labeled SMS messages, classified as spam or ham, and is used to train the model to make accurate predictions.

The model is deployed on [ https://sms-spam-classifier-1-armw.onrender.com ] and is ready for you to test. Just enter an SMS message into the web interface, and the model will classify it in real-time.
