import tkinter as tk
from tkinter import messagebox
import joblib

# Function to predict credit category
def predict_credit():
    # Load the trained model
    model = joblib.load('trained_model.pkl')

    # Extract user inputs from entry fields
    duration_month = int(entry_duration_month.get())
    credit_history = entry_credit_history.get()
    purpose = entry_purpose.get()
    savings_acc = entry_savings_acc.get()
    employment_st = int(entry_employment_st.get())
    poi = int(entry_poi.get())
    personal_status = entry_personal_status.get()
    guarantors = entry_guarantors.get()
    resident_since = int(entry_resident_since.get())
    property_type = entry_property_type.get()
    age = int(entry_age.get())
    installment_type = entry_installment_type.get()
    housing_type = entry_housing_type.get()
    credits_no = int(entry_credits_no.get())
    job_type = entry_job_type.get()
    liables = int(entry_liables.get())
    telephone = entry_telephone.get()
    foreigner = entry_foreigner.get()

    # Prepare user inputs as a DataFrame (similar to the training data)
    customer_info = [[duration_month, credit_history, purpose, savings_acc, employment_st,
                      poi, personal_status, guarantors, resident_since, property_type,
                      age, installment_type, housing_type, credits_no, job_type,
                      liables, telephone, foreigner]]

    # Make prediction using the loaded model
    prediction = model.predict(customer_info)[0]

    # Display prediction
    messagebox.showinfo("Prediction", f"Predicted Credit Category: {prediction}")

# Create main application window
root = tk.Tk()
root.title("Credit Category Predictor")

# Add widgets (entry fields for user input)
# Example: Duration Month
label_duration_month = tk.Label(root, text="Duration Month:")
label_duration_month.grid(row=0, column=0)
entry_duration_month = tk.Entry(root)
entry_duration_month.grid(row=0, column=1)

# Repeat similar steps for other entry fields...

# Add button to trigger prediction
predict_button = tk.Button(root, text="Predict Credit Category", command=predict_credit)
predict_button.grid(row=18, columnspan=2)

# Run the main event loop
root.mainloop()
