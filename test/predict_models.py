# predict_models.py
import joblib
import pandas as pd
import numpy as np

# ---------------------------
# 1. Load trained models
# ---------------------------
rf_pipeline = joblib.load("rf_lie_model.pkl")

# ---------------------------
# 2. Load new dataset
# ---------------------------
df_new = pd.read_excel("edit.xlsx")

# Drop unused columns (if present)
df_new = df_new.drop(columns=['Audio','Start','End','Speaker'], errors='ignore')

# ---------------------------
# 3. Rename columns to match training
# ---------------------------
df_new.rename(columns={
    'Amplitude(Shimmer)': 'Shimmer_dB',
    'Frequency(Jitter)': 'Jitter_RAP',
    'Harmonicity': 'HNR',
    # Add more renames if your extractor uses different names
}, inplace=True)

# ---------------------------
# 4. Ensure numeric columns
# ---------------------------
for col in df_new.columns:
    if col != 'Label':
        df_new[col] = pd.to_numeric(df_new[col], errors='coerce')

# ---------------------------
# 5. Features expected by trained model
# ---------------------------
trained_features = [
    'Pitch_Median', 'Pitch_Mean', 'Pitch_Max', 'Pitch_Min',
    'Pitch_Std', 'Jitter_Local', 'Jitter_RAP', 'Jitter_PPQ5',
    'Shimmer_Local', 'Shimmer_dB', 'Shimmer_APQ3',
    'HNR', 'NHR', 'Unvoiced_Frames', 'Voice_Breaks',
    'Gender'  # this was missing and caused the error
]

# ---------------------------
# 6. Add missing columns with default values
# ---------------------------
for col in trained_features:
    if col not in df_new.columns:
        print(f"Warning! Missing column '{col}' added with default value 0")
        df_new[col] = 0

# ---------------------------
# 7. Select features
# ---------------------------
X_new = df_new[trained_features]

# ---------------------------
# 8. Make predictions
# ---------------------------
rf_pred = rf_pipeline.predict(X_new)

# ---------------------------
# 9. Print results
# ---------------------------
print("Random Forest Predictions:", rf_pred)