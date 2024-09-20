from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
data_path = 'Emotion_final_with_predictions.csv'  # Ensure this path is correct for deployment
data = pd.read_csv(data_path)

# Prepare data
X = data['Text']
y = data['Emotion']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize text data
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train Logistic Regression model
logistic_regression_model = LogisticRegression(max_iter=1000)
logistic_regression_model.fit(X_train_vec, y_train)

# Make predictions
y_pred = logistic_regression_model.predict(X_test_vec)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f'Logistic Regression Accuracy: {accuracy:.4f}')
print('Classification Report:')
print(classification_report(y_test, y_pred))

# Prediction function
def predict_emotion_logistic(text):
    text_vec = vectorizer.transform([text])
    return logistic_regression_model.predict(text_vec)[0]

# Streamlit app
st.title('Emotion Prediction using Logistic Regression')
st.subheader('Dataset')
st.write(data.head())

# Input for prediction
st.subheader('Predict Emotion from Text')
input_text = st.text_input('Enter text:')

if st.button('Predict'):
    if input_text:
        predicted_emotion = predict_emotion_logistic(input_text)
        st.write(f'Predicted Emotion: {predicted_emotion}')
    else:
        st.write('Please enter some text to predict the emotion.')
