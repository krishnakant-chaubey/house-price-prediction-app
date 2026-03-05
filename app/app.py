from flask import Flask, request, render_template
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    features = [float(x) for x in request.form.values()]
    final_features = np.array([features])
    prediction = model.predict(final_features)

    return render_template(
        "index.html",
        prediction_text=f"Predicted Price: {prediction[0]}"
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
