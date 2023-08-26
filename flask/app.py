from flask import Flask, render_template, request
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import train_and_save_model as Tmodel


app = Flask(__name__)

# Load your trained model and vectorizer using pickle
with open('trained_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Load your data (replace with your actual data loading code)
data = pd.read_csv('Mental_Health_FAQ.csv', delimiter=',')

# Preprocess the questions for similarity matching
data['Processed_Questions'] = data['Questions'].apply(lambda x: x.lower())

# Define the home page route
@app.route('/')
def home():
    return render_template('home.html')  # Render the home page template

# Define the route to handle user inputs
@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form['user_input']
    
    response = Tmodel.user_input(user_input)

    return render_template('chat.html', user_input=user_input, response=response)

if __name__ == '__main__':
    app.run(debug=True)
