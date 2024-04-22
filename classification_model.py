import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Step 1: Load the training dataset
train_data = pd.read_csv('C_T_train_dataset.csv')

# Step 3: Split the dataset into features (X_train) and target variable (y_train)
X_train = train_data.drop('Group_no', axis=1)
y_train = train_data['Group_no']

# Step 4: One-hot encode categorical variables
X_train_encoded = pd.get_dummies(X_train)

# Step 5: Choose a machine learning algorithm and train the model
model = RandomForestClassifier()
model.fit(X_train_encoded, y_train)

# Step 6: Load the test dataset
test_data = pd.read_csv('C_T_test_dataset.csv')

# Step 7: Preprocess the test dataset (one-hot encoding)
test_data_encoded = pd.get_dummies(test_data)

# Step 8: Use the trained model to make predictions on the test dataset
predictions = model.predict(test_data_encoded)

# Assuming Sno is the first column in the test data
sno_column = test_data.iloc[:, 0]  # Assuming the first column is Sno
sno_column = sno_column.rename("Sno")  # Rename the series to "Sno"

# Convert predictions to a DataFrame
predictions_df = pd.DataFrame(predictions, columns=["Group_no"])

# Concatenate Sno and predictions DataFrames
output_df = pd.concat([sno_column, predictions_df], axis=1)

# Save the output to a CSV file
output_df.to_csv("predictions_output.csv", index=False)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)
