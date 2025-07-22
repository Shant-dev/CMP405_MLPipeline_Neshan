
import numpy as np
import joblib

class RevenueAgent:
    def __init__(self, model_path='rf_model.pkl', encoders_path='label_encoders.pkl', target_encoder_path='target_encoder.pkl'):
        self.model = joblib.load(model_path)
        self.label_encoders = joblib.load(encoders_path)
        self.target_encoder = joblib.load(target_encoder_path)

    def preprocess_input(self, input_dict):
        processed = {}
        for key in input_dict:
            if key in self.label_encoders:
                processed[key] = self.label_encoders[key].transform([input_dict[key]])[0]
            else:
                processed[key] = input_dict[key]
        return np.array(list(processed.values())).reshape(1, -1)

    def predict(self, input_dict):
        X = self.preprocess_input(input_dict)
        prediction = self.model.predict(X)[0]
        probability = self.model.predict_proba(X)[0][1]
        class_label = self.target_encoder.inverse_transform([prediction])[0]
        return {
            "prediction": class_label,
            "confidence": round(probability, 2),
            "action": "Send offer email" if prediction else "No action needed"
        }

# Create the agent
agent = RevenueAgent()

# Define the expected input fields and their types
input_fields = {
    'Administrative': int,
    'Administrative_Duration': float,
    'Informational': int,
    'Informational_Duration': float,
    'ProductRelated': int,
    'ProductRelated_Duration': float,
    'BounceRates': float,
    'ExitRates': float,
    'PageValues': float,
    'SpecialDay': float,
    'Month': str,
    'OperatingSystems': int,
    'Browser': int,
    'Region': int,
    'TrafficType': int,
    'VisitorType': str,
    'Weekend': str
}

# Ask user for input
print("🔎 Please enter the following values for prediction:")

user_input = {}
for field, dtype in input_fields.items():
    while True:
        value = input(f"Enter value for '{field}' ({dtype.__name__}): ")
        try:
            # Clean string inputs for certain fields
            if dtype == str:
                user_input[field] = value.strip()
            else:
                user_input[field] = dtype(value)
            break
        except ValueError:
            print(f"❌ Invalid input. Please enter a valid {dtype.__name__} for {field}.")

# Make prediction
result = agent.predict(user_input)

# Output result
print("\n✅ Prediction Result:")
print("🎯 Prediction:", result["prediction"])
print("📊 Confidence:", result["confidence"])
print("📩 Recommended Action:", result["action"])
