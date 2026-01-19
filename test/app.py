from flask import Flask, render_template, request, jsonify
import os
import uuid
import joblib

from extract_features import extract_features

# -------------------------
# Flask App Initialization
# -------------------------
app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -------------------------
# Load ML Model (ONCE)
# -------------------------
MODEL_PATH = "rf_lie_model.pkl"
model = joblib.load(MODEL_PATH)

# -------------------------
# ROUTES (PAGES)
# -------------------------

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/second")
def second():
    return render_template("second.html")


@app.route("/live")
def live_audio():
    return render_template("liveAudio.html")


# -------------------------
# FILE UPLOAD PREDICTION
# -------------------------
@app.route("/predict", methods=["POST"])
def predict():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    audio = request.files["audio"]

    if audio.filename == "":
        return jsonify({"error": "Empty file"}), 400

    # Save audio with unique name
    filename = f"{uuid.uuid4()}.wav"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    audio.save(filepath)

    try:
        # Extract features
        features_df = extract_features(filepath)

        # Predict
        prediction = model.predict(features_df)[0]
        probabilities = model.predict_proba(features_df)[0]

        return jsonify({
            "result": "Lie" if prediction == 1 else "Truth",
            "truth_probability": float(probabilities[0]),
            "lie_probability": float(probabilities[1])
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -------------------------
# LIVE AUDIO PREDICTION
# -------------------------
@app.route("/predict-live", methods=["POST"])
def predict_live():
    if "audio" not in request.files:
        return jsonify({"error": "No live audio received"}), 400

    audio = request.files["audio"]

    filename = f"live_{uuid.uuid4()}.wav"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    audio.save(filepath)

    try:
        features_df = extract_features(filepath)

        prediction = model.predict(features_df)[0]
        probabilities = model.predict_proba(features_df)[0]

        return jsonify({
            "result": "Lie" if prediction == 1 else "Truth",
            "truth_probability": float(probabilities[0]),
            "lie_probability": float(probabilities[1])
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":
    app.run(debug=True)
