# train_and_save_model.py
# Import required libraries
import pickle
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import plotly.express as px
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


# Load the dataset
data = pd.read_csv('Mental_Health_FAQ.csv', delimiter=',')


# Preprocess the data
data['Questions'] = data['Questions'].str.lower()
data['Questions'] = data['Questions'].str.replace('[^\w\s]', '')
data = data.dropna()

# Exploratory Data Analysis
# intent_counts = data['Questions'].value_counts()
# fig = px.bar(x=intent_counts.index, y=intent_counts.values,
#              labels={'x': 'Intents', 'y': 'Count'})
# fig.show()

# Questions and Response Analysis
data['Intent'] = data['Questions']
# questions_response_counts = data.groupby(
#     'Intent').size().reset_index(name='Count')
# avg_questions = data.groupby('Intent').size().mean()
# fig = px.bar(x=questions_response_counts['Intent'], y=questions_response_counts['Count'],
#              labels={'x': 'Intents', 'y': 'Average Questions Count'})
# fig.show()

# Intent Prediction Model
X_train, X_test, y_train, y_test = train_test_split(data['Questions'], data['Intent'], test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LinearSVC()
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)
report = classification_report(y_test, y_pred)
print(report)

# # Visualize model performance
# metrics = ['precision', 'recall', 'f1-score', 'support']
# scores = classification_report(y_test, y_pred, output_dict=True)[
#     'weighted avg']
# fig = px.bar(x=metrics, y=[scores[metric] for metric in metrics], labels={
#              'x': 'Metrics', 'y': 'Score'})
# # fig.show()


# # Save the trained model and vectorizer using pickle
# try:
#     with open('trained_model.pkl', 'wb') as model_file:
#         pickle.dump(model, model_file)

#     with open('vectorizer.pkl', 'wb') as vectorizer_file:
#         pickle.dump(vectorizer, vectorizer_file)
#     print("Model and vectorizer saved successfully.")
# except Exception as e:
#     print("An error occurred while saving:", e)

# # Prediction Model Deployment
# print("Welcome to the Mental Health FAQ Chatbot!")
# print("Ask a question or enter 'quit' to exit.")

# while True:
#     user_input = input("User: ")

#     if user_input.lower() == 'quit':
#         print("Chatbot: Goodbye!")
#         break

#     user_input_vec = vectorizer.transform([user_input.lower()])
#     predicted_intent = model.predict(user_input_vec)[0]

#     response = data[data['Questions'] == predicted_intent]['Answers'].values[0] if predicted_intent in data[
#         'Questions'].values else "I'm sorry, I don't have a response for that question."

#     print("Chatbot: " + response)
# Save the trained model and vectorizer using pickle

def user_input(input):
     
    if input.lower() == 'quit':
        return "Chatbot: Goodbye!"

    user_input_vec = vectorizer.transform([user_input.lower()])
    predicted_intent = model.predict(user_input_vec)[0]

    return (
        data[data['Questions'] == predicted_intent]['Answers'].values[0]
        if predicted_intent in data['Questions'].values
        else "I'm sorry, I don't have a response for that question."
    )


