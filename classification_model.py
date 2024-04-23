import pandas as pd
from sklearn.ensemble import RandomForestClassifier

train_data = pd.read_csv('C_T_train_dataset.csv')

X_train = train_data.drop('Group_no', axis=1)
y_train = train_data['Group_no']

X_train_encoded = pd.get_dummies(X_train)

model = RandomForestClassifier()
model.fit(X_train_encoded, y_train)

test_data = pd.read_csv('C_T_test_dataset.csv')

test_data_encoded = pd.get_dummies(test_data)

predictions = model.predict(test_data_encoded)

sno_column = test_data.iloc[:, 0]
sno_column = sno_column.rename("Sno")

predictions_df = pd.DataFrame(predictions, columns=["Group_no"])

output_df = pd.concat([sno_column, predictions_df], axis=1)

output_df.to_csv("predictions_output.csv", index=False)
