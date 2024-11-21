from flask import Flask, render_template, request
import pandas as pd
import joblib

# Initialize the Flask app
app = Flask(__name__)

# Load the trained Naive Bayes model and label encoder
nb_model_filename = 'models/naive_bayes_model.pkl'
le_filename = 'models/label_encoder.pkl'

# Load the Naive Bayes model and label encoder using joblib
nb_model = joblib.load(nb_model_filename)
label_encoder = joblib.load(le_filename)

# Load any scaler if used during training (optional, if scaling was used during training)
scaler_filename = 'models/scaler.pkl'
try:
    with open(scaler_filename, 'rb') as file:
        scaler = joblib.load(file)
except FileNotFoundError:
    scaler = None  # If no scaler file is present, just skip scaling

@app.route('/')
def home():
    return render_template('home.html')  # The form where users input data for prediction

@app.route('/naive_bayes', methods=['GET', 'POST'])
def naive_bayes():
    if request.method == 'POST':
        try:
            # Retrieve form data for 'Income' and 'SpendingScore'
            income = float(request.form['income'])
            spending_score = float(request.form['spending_score'])

            # Validate input data
            if income <= 0 or spending_score < 0 or spending_score > 100:
                return render_template('naive_bayes.html', error="Invalid input values. Please ensure that income is positive and spending score is between 0 and 100.")

            # Prepare data for prediction
            input_data = pd.DataFrame({
                'Income': [income],
                'SpendingScore': [spending_score]
            })

            # If a scaler was used during training, scale the data
            if scaler:
                input_data = scaler.transform(input_data)

            # Make prediction using the Naive Bayes model
            predicted_class = nb_model.predict(input_data)[0]
            
            # Convert the numerical prediction back to the actual label (e.g., Low, Medium, High)
            predicted_label = label_encoder.inverse_transform([predicted_class])[0]

            # Return the result to the user
            return render_template(
                'naive_bayes.html',
                prediction=predicted_label,
                income=income,
                spending_score=spending_score
            )

        except Exception as e:
            return render_template('naive_bayes.html', error=f"An error occurred: {str(e)}")

    return render_template('naive_bayes.html')

if __name__ == '__main__':
    app.run(debug=True)
