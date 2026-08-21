

import os
import joblib


FOLDER_PATH = os.path.dirname(os.path.dirname(__file__))

# Load artifacts
preprocessor = joblib.load(
    os.path.join(FOLDER_PATH, "artifacts", "preprocessor.joblib")
)

xgb_model =joblib.load(
    os.path.join(FOLDER_PATH, "artifacts", "best_model.keras")
)