import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB

# Step 1: Load the dataset
file_path = 'customer_segmentation_dataset.csv'  # Update this if needed
data = pd.read_csv(file_path)

# Step 2: Preprocess the data
# Drop the 'CustomerID' column as it's not needed for training
# If you need to keep it for customer identification, you can keep it in the original data
data = data.drop(columns=["CustomerID"])

# Use only 'Income' and 'SpendingScore' as features
X = data[['Income', 'SpendingScore']]  # Features: Income and SpendingScore
y = data['PurchasingHabit']  # Target: PurchasingHabit

# Encode the target variable ('PurchasingHabit')
le = LabelEncoder()
y = le.fit_transform(y)  # Encode 'Low', 'Medium', 'High' as numerical values

# Step 3: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the Naive Bayes model
model = GaussianNB()
model.fit(X_train, y_train)

# Step 5: Make predictions on the test set
y_pred = model.predict(X_test)

# Step 6: Convert numerical predictions back to original labels
predicted_labels = le.inverse_transform(y_pred)

# Step 7: Combine the predictions with 'Income' and 'SpendingScore' into a DataFrame
output_df = X_test.copy()  # Copy 'Income' and 'SpendingScore' to the new DataFrame
output_df['PredictedPurchasingHabit'] = predicted_labels  # Add the predictions as a new column

# Step 8: Add a "Customer" column with sequential customer IDs (or you can use original Customer IDs if available)
output_df['Customer'] = range(1, len(output_df) + 1)  # Adding sequential customer numbers starting from 1

# Step 9: Display the results
print(output_df[['Customer', 'Income', 'SpendingScore', 'PredictedPurchasingHabit']].head())  # Show the first few rows
