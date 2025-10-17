from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load model and columns
model = joblib.load("models/model.joblib")
model_columns = joblib.load("models/model_columns.joblib")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form data safely
        area = float(request.form.get("area"))
        bedrooms = int(request.form.get("bedrooms"))
        baths = int(request.form.get("baths"))
        city = request.form.get("city")

        # Build input data
        input_data = pd.DataFrame([{
            "area": area,
            "bedrooms": bedrooms,
            "baths": baths,
            "city": city
        }])

        # One-hot encode city
        all_cities = ["Islamabad", "Karachi", "Lahore", "Rawalpindi"]
        for c in all_cities:
            input_data[f"city_{c}"] = 1 if city == c else 0
        input_data.drop(columns=["city"], inplace=True)

        # Align columns with training data
        input_data = input_data.reindex(columns=model_columns, fill_value=0)

        # Predict
        prediction = model.predict(input_data)[0]
        return render_template("index.html", prediction_text=f"🏠 Predicted Price: {prediction:.2f}")

    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/login")
def login():
    return "Login page for users"

@app.route("/dashboard")
def dashboard():
    return "Dashboard page for users"
if __name__ == "__main__":
    app.run(debug=True)
