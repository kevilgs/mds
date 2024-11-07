from flask import Flask, request, render_template, redirect, url_for, session, flash
import joblib
import pandas as pd
import numpy as np
from database import add_user, get_user, add_user_info

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Load the trained model
model = joblib.load('model/logistic_regression_model.pkl')

# Load the symptom encoder
symptom_encoder = joblib.load('model/symptom_encoder.pkl')

# Load the disease encoder
disease_encoder = joblib.load('model/disease_encoder.pkl')

# Load the column names used during training
with open('model/columns.pkl', 'rb') as f:
    columns = joblib.load(f)

# Load the symptoms from a file
symptoms = pd.read_csv('data/extracted_symptoms.csv', header=None)[0].tolist()

@app.route('/')
def home():
    if 'user_id' in session:
        return redirect(url_for('welcome'))
    return redirect(url_for('login'))

@app.route('/welcome')
def welcome():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('Welcome.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = get_user(username)
        if user and user[2] == password:
            session['user_id'] = user[0]
            session['username'] = user[1]
            return redirect(url_for('welcome'))
        else:
            flash('Invalid username or password')
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        name = request.form.get('name')
        email = request.form.get('email')
        add_user(username, password, name, email)
        flash('Account created successfully. Please log in.')
        return redirect(url_for('login'))
    return render_template('login.html')

@app.route('/information', methods=['GET', 'POST'])
def information():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    if request.method == 'POST':
        age = request.form.get('age')
        gender = request.form.get('gender')
        session['age'] = age
        session['gender'] = gender
        return redirect(url_for('symptoms_page'))
    return render_template('Information.html')

@app.route('/symptoms', methods=['GET', 'POST'])
def symptoms_page():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    if request.method == 'POST':
        # Get symptoms and additional information from the form
        symptoms_input = request.form.get('symptoms', '').split(', ')
        age = request.form.get('age')
        overweight = request.form.get('overweight')
        smoke = request.form.get('smoke')
        injured = request.form.get('injured')
        cholesterol = request.form.get('cholesterol')
        hypertension = request.form.get('hypertension')
        diabetes = request.form.get('diabetes')

        # Validate symptoms
        valid_symptoms = [symptom for symptom in symptoms_input if symptom in symptoms]
        if not valid_symptoms:
            return render_template('Symptoms.html', symptoms=symptoms, error="Invalid symptoms provided.")

        # Encode symptoms as done during training
        try:
            symptom_encoded = symptom_encoder.transform(valid_symptoms)
        except ValueError as e:
            return render_template('Symptoms.html', symptoms=symptoms, error=f"Error encoding symptoms: {e}")

        symptom_df = pd.DataFrame(symptom_encoded, columns=['Symptom_Encoded'])
        symptom_df = pd.get_dummies(symptom_df['Symptom_Encoded'])

        # Add additional features to the DataFrame
        feature_mapping = {'yes': 1, 'no': 0, 'dont_know': 2}
        additional_features = {
            'Overweight': feature_mapping[overweight],
            'Smoke': feature_mapping[smoke],
            'Injured': feature_mapping[injured],
            'Cholesterol': feature_mapping[cholesterol],
            'Hypertension': feature_mapping[hypertension],
            'Diabetes': feature_mapping[diabetes]
        }
        for feature, value in additional_features.items():
            symptom_df[feature] = value

        # Align the dummy variables with the training columns
        symptom_df = symptom_df.reindex(columns=columns, fill_value=0)

        # Make prediction
        probabilities = model.predict_proba(symptom_df)[0]
        top_indices = np.argsort(probabilities)[-3:][::-1]
        top_diseases = disease_encoder.inverse_transform(top_indices)
        top_probabilities = probabilities[top_indices]

        results = list(zip(top_diseases, top_probabilities))

        # Store user information and prediction in the database
        user_id = session.get('user_id')
        if user_id:
            add_user_info(user_id, age, overweight, smoke, injured, cholesterol, hypertension, diabetes, ', '.join(valid_symptoms), top_diseases[0])

        return render_template('Result.html', results=results)

    return render_template('Symptoms.html', symptoms=symptoms)

@app.route('/result')
def result():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('Result.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)