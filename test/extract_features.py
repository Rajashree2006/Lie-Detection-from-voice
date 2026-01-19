import librosa
import numpy as np
import pandas as pd
import joblib


MODEL_PATH = "rf_lie_model.pkl"   # source of truth for features


def get_expected_features():
    """
    Load trained model and extract expected feature names
    """
    model = joblib.load(MODEL_PATH)

    for step in model.named_steps.values():
        if hasattr(step, "feature_names_in_"):
            return list(step.feature_names_in_)

    raise RuntimeError("Could not find feature_names_in_ in trained pipeline")


def extract_audio_features(audio_path: str) -> dict:
    """
    Extract RAW audio features (must match training logic)
    """
    y, sr = librosa.load(audio_path, sr=None)

    features = {}

    # -------------------
    # Pitch (F0)
    # -------------------
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7")
    )
    f0 = f0[~np.isnan(f0)]

    features["Pitch_Mean"] = np.mean(f0) if len(f0) else 0
    features["Pitch_Median"] = np.median(f0) if len(f0) else 0
    features["Pitch_Min"] = np.min(f0) if len(f0) else 0
    features["Pitch_Max"] = np.max(f0) if len(f0) else 0
    features["Pitch_Std"] = np.std(f0) if len(f0) else 0

    # -------------------
    # Energy
    # -------------------
    rms = librosa.feature.rms(y=y)[0]
    features["Energy_Mean"] = np.mean(rms)
    features["Energy_Std"] = np.std(rms)

    # -------------------
    # Spectral Features
    # -------------------
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]

    features["SpectralCentroid_Mean"] = np.mean(centroid)
    features["SpectralBandwidth_Mean"] = np.mean(bandwidth)

    # -------------------
    # Zero Crossing Rate
    # -------------------
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    features["ZCR_Mean"] = np.mean(zcr)

    return features


def extract_features(audio_path: str) -> pd.DataFrame:
    """
    MAIN ENTRY POINT
    Returns a DataFrame that EXACTLY matches training features
    """
    raw_features = extract_audio_features(audio_path)
    expected_features = get_expected_features()

    # Build aligned feature vector
    aligned = {}

    for feature in expected_features:
        aligned[feature] = raw_features.get(feature, 0.0)

    return pd.DataFrame([aligned])
