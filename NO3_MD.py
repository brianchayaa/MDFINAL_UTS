# inference.py
import pandas as pd
import pickle

class LoanDefaultPredictor:
    def __init__(self, model_path="loan_model.pkl"):
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

    def predict(self, input_data):
        df = pd.DataFrame([input_data])
        prediction = self.model.predict(df)[0]
        return "Approved" if prediction == 1 else "Denied"

if __name__ == "__main__":
    predictor = LoanDefaultPredictor()

    # Contoh input (HARUS sesuai kolom & format training data)
    sample_input = {
        "person_age": 30,
        "person_gender": "male",
        "person_education": "Bachelor",
        "person_income": 60000,
        "person_emp_exp": 5,
        "person_home_ownership": "RENT",
        "loan_amnt": 15000,
        "loan_intent": "PERSONAL",
        "loan_int_rate": 12.5,
        "loan_percent_income": 0.25,
        "cb_person_cred_hist_length": 5,
        "credit_score": 700,
        "previous_loan_defaults_on_file": "No"
    }

    result = predictor.predict(sample_input)
    print(f"Loan prediction: {result}")