import pickle
from flask import Flask, jsonify, request
import numpy as np

app = Flask(__name__)

# Load the trained SVM model
MODEL_PATH = "tgs svm.pkl"
try:
  with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)
  print("Model loaded successfully.")
except Exception as e:
  print(f"Error loading model: {e}")
  model = None


@app.route("/", methods=["GET"])
def home():
  return jsonify({
      "status": "online",
      "message": (
          "SVM Model API is running. Send a POST request to /predict with JSON"
          " data."
      ),
  })


@app.route("/predict", methods=["POST"])
def predict():
  if model is None:
    return jsonify({"error": "Model not loaded on server."}), 500

  try:
    data = request.get_json()

    # Define the expected feature keys in the exact order the model expects
    feature_keys = [
        "age",
        "gender",
        "city",
        "bmi",
        "family_history_diabetes",
        "physical_activity_level",
        "diet_type",
        "smoking_status",
        "alcohol_consumption",
        "hours_sleep_per_night",
        "stress_level",
        "fasting_blood_sugar",
        "hba1c_level",
        "blood_pressure_systolic",
        "blood_pressure_diastolic",
        "waist_circumference_cm",
        "income_bracket",
    ]

    # Extract features from incoming JSON request
    # Note: If your categorical columns require encoding (like One-Hot or Label Encoding),
    # ensure they are pre-processed to numerical format before passing to the model.
    features = [data.get(key) for key in feature_keys]

    if None in features:
      return (
          jsonify({
              "error": (
                  "Missing one or more required feature fields in JSON input."
              ),
              "expected_fields": feature_keys,
          }),
          400,
      )

    # Convert to numpy array and reshape for prediction
    input_data = np.array([features], dtype=float)
    prediction = model.predict(input_data)

    # Convert prediction result to native python type for JSON serialization
    result = int(prediction[0])

    return jsonify({"prediction": result})

  except Exception as e:
    return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
  app.run(host="0.0.0.0", port=5000)
