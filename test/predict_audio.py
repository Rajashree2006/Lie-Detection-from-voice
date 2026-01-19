import joblib
from extract_features import extract_features

# Load trained model
model = joblib.load("rf_lie_model.pkl")

# Extract features
features = extract_features("test_audio.wav")

# EXACT feature order used during training
trained_features = [
    'Pitch_Median', 'Pitch_Mean', 'Pitch_Max', 'Pitch_Min',
    'Pitch_Std',
    'Jitter_Local', 'Jitter_RAP', 'Jitter_PPQ5',
    'Shimmer_Local', 'Shimmer_dB', 'Shimmer_APQ3',
    'HNR', 'NHR',
    'Unvoiced_Frames', 'Voice_Breaks'
]

features = features[trained_features]

# Predict
prediction = model.predict(features)
probability = model.predict_proba(features)

label = "Truth" if prediction[0] == 1 else "Lie"

print("Prediction:", label)
print("Confidence [Lie, Truth]:", probability)
