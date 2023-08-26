# app.py

from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load your trained model and vectorizer using pickle
with open('trained_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Load your data (replace with your actual data loading code)
data = pd.read_csv('Mental_Health_FAQ.csv', delimiter=',')

# Define the home page route


@app.route('/')
def home():
    return render_template('home.html')  # Render the home page template

# Define the route to handle user inputs


@app.route('/chat', methods=['POST'])
def chat():

    user_input = request.form['user_input']

    user_input_vec = vectorizer.transform([user_input.lower()])
    predicted_intent = model.predict(user_input_vec)[0]

    response = data[data['Questions'] == predicted_intent]['Answers'].values[0] if predicted_intent in data[
        'Questions'].values else "I'm sorry, I don't have a response for that question."

    return render_template('chat.html', user_input=user_input, response=response)


if __name__ == '__main__':
    app.run(debug=True)

app.run(host='0.0.0.0', port=81)
