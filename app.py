from flask import Flask, render_template, request
import pandas as pd
import joblib
import pickle

app = Flask(__name__)

# Load the trained models and encoders
model_filename = 'models/retailwise_model.pkl'
nb_model_filename = 'models/naive_bayes_model.pkl'
le_filename = 'models/label_encoder.pkl'
knn_model_filename = 'models/product_category_model.pkl'
le_brand_filename = 'models/brand_name_encoder.pkl'
le_category_filename = 'models/product_category_encoder.pkl'

# Load the models and encoders using joblib and pickle
with open(model_filename, 'rb') as file:
    model = pickle.load(file)

with open(nb_model_filename, 'rb') as file:
    nb_model = joblib.load(file)

label_encoder = joblib.load(le_filename)
label_encoder_brand = joblib.load(le_brand_filename)
label_encoder_category = joblib.load(le_category_filename)
knn_model = joblib.load(knn_model_filename)

scaler_filename = 'models/scaler.pkl'
try:
    with open(scaler_filename, 'rb') as file:
        scaler = joblib.load(file)
except FileNotFoundError:
    scaler = None  # If no scaler file is present, just skip scaling

recommendations = {
    'Beverages': ['Coca-Cola', 'Sprite', 'C2'],
    'Snacks': ['Oreo', 'KitKat', 'Lay\'s'],
    'Dairy': ['Nestle Milk', 'Milo', 'Cheese'],
    'Bakery': ['Bread', 'Butter', 'Croissant'],
    'Produce': ['Fruits', 'Vegetables', 'Juice']
}
def get_recommendations(product_category):
    return recommendations.get(product_category, ['No recommendations available'])

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/linear_regression', methods=['GET', 'POST'])
def linear_regression():
    if request.method == 'POST':
        try:
            # Retrieve form data
            annual_income = float(request.form['annual_income'])
            marketing_spend = float(request.form['marketing_spend'])
            competitor_price = float(request.form['competitor_price'])

            # Prepare data for prediction
            input_data = pd.DataFrame({
                'AnnualIncome': [annual_income],
                'MarketingSpend': [marketing_spend],
                'CompetitorPrice': [competitor_price]
            })

            # Make prediction
            predicted_sales = model.predict(input_data)[0]

            return render_template(
                'linear_regression.html',
                prediction=round(predicted_sales, 2),
                annual_income=annual_income,
                marketing_spend=marketing_spend,
                competitor_price=competitor_price
            )
        except Exception as e:
            return render_template('linear_regression.html', error=str(e))
    return render_template('linear_regression.html')

# Placeholder routes for other algorithms
@app.route('/decision_tree')
def decision_tree():
    return render_template('decision_tree.html')

@app.route('/knn', methods=['GET', 'POST'])
def knn():
    if request.method == 'POST':
        try:
            # Retrieve form data for Brand Name
            brand_name = request.form['brand_name']

            # Validate input data
            if not brand_name:
                return render_template('knn.html', error="Invalid input. Please provide a brand name.")

            # Encode the brand name for prediction
            encoded_brand = label_encoder_brand.transform([brand_name])[0]

            # Make prediction using the KNN model
            predicted_category_class = knn_model.predict([[encoded_brand]])[0]
            
            # Decode the predicted product category
            predicted_category = label_encoder_category.inverse_transform([predicted_category_class])[0]

            # Get recommendations based on the predicted category
            recommended_products = get_recommendations(predicted_category)

            # Return the result to the user
            return render_template(
                'knn.html',
                prediction=predicted_category,
                brand_name=brand_name,
                recommendations=recommended_products
            )

        except Exception as e:
            return render_template('knn.html', error=f"An error occurred: {str(e)}")

    return render_template('knn.html')

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

@app.route('/neural_network')
def neural_network():
    return render_template('neural_network.html')

@app.route('/svm')
def svm():
    return render_template('svm.html')

if __name__ == '__main__':
    app.run(debug=True)
