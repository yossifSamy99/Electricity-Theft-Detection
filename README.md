# ⚡ Electricity & Gas Theft Detection

A machine learning system that classifies hourly electricity/gas consumption
readings from commercial buildings as **Normal** or one of **6 theft
patterns** (`Theft1`–`Theft6`), using an XGBoost multi-class classifier
served through a self-contained Streamlit UI.

---

## 📌 Overview

Utility meter readings (electricity + gas, broken down by end-use: fans,
cooling, heating, interior lighting/equipment, water heating) are engineered
into time-cyclical and consumption-ratio features, transformed with a
per-column Box-Cox / Yeo-Johnson power transform, and fed into an XGBoost
classifier trained to distinguish normal usage from known theft/fraud
signatures.

The Streamlit app loads the trained model and preprocessing pipeline
**directly via `joblib`** — there is no separate API layer in this version.

---

## 🗂️ Project Structure

```
.
├── app.py                          # Streamlit UI (single + batch prediction)
├── final_notebook.ipynb            # EDA, feature engineering, model training
├── requirements.txt
├── data/
│   └── df.csv                      # Training dataset
├── src/
│   ├── artifacts/
│   │   ├── best_xgboost_model.pkl  # Trained XGBoost model
│   │   └── preprocessor.joblib     # Fitted sklearn Pipeline (feature eng. + encoding)
│   └── utils/
│       ├── transformers.py         # Custom sklearn transformers (FeatureEngineer, MixedPowerTransformer)
│       └── schema.py               # Column names, building classes, theft labels
└── test/
```

> ⚠️ **`src/utils/transformers.py` is not optional.** The saved
> `preprocessor.joblib` is a pickled `sklearn.Pipeline` containing custom
> transformer classes. `joblib.load()` can only unpickle them if this exact
> module is importable at the same path — it must ship with the model, and
> the notebook must import the classes from here (not redefine them inline)
> before calling `.fit()` and saving.

---

## ⚙️ Installation

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
```

**Pin your scikit-learn version explicitly.** A `preprocessor.joblib` fitted
under one scikit-learn version can fail to unpickle under another
(e.g. internal classes like `ColumnTransformer`'s remainder-columns helper
have changed between minor versions). Train and serve with the *same*
version:

```
scikit-learn==1.6.1
xgboost==2.1.1
streamlit==1.38.0
pandas
numpy
plotly
joblib
```

---

## ▶️ Usage

```bash
streamlit run app.py
```

Place `best_xgboost_model.pkl` and `preprocessor.joblib` in
`src/artifacts/` before launching. The app has two tabs:

- **🔍 Single Prediction** — manual form for one reading (electricity/gas
  values, building type, date + hour). Returns the predicted class with a
  confidence score and a per-class probability chart.
- **📁 Batch Prediction (CSV)** — upload a CSV with the same schema as the
  training data; get predictions, a class-distribution summary, and a
  downloadable results CSV.

A ready-to-use `sample_test_batch.csv` is included for quickly testing the
batch tab end-to-end.

---

## 🧾 Data Schema

**Raw input columns** (required for both single and batch prediction):

| Column | Description |
|---|---|
| `row_id` | Sequential reading index (used to derive time-of-day / day-of-week / month features) |
| `Electricity:Facility [kW](Hourly)` | Total electricity load |
| `Fans:Electricity [kW](Hourly)` | Fan electricity load |
| `Cooling:Electricity [kW](Hourly)` | Cooling electricity load |
| `Heating:Electricity [kW](Hourly)` | Heating electricity load |
| `InteriorLights:Electricity [kW](Hourly)` | Interior lighting load |
| `InteriorEquipment:Electricity [kW](Hourly)` | Interior equipment load |
| `Gas:Facility [kW](Hourly)` | Total gas load |
| `Heating:Gas [kW](Hourly)` | Gas heating load |
| `InteriorEquipment:Gas [kW](Hourly)` | Gas-powered equipment load |
| `Water Heater:WaterSystems:Gas [kW](Hourly)` | Water heater gas load |
| `Class` | Building type (see list below) |

**Building classes (`Class`):** FullServiceRestaurant, Hospital,
LargeHotel, LargeOffice, MediumOffice, MidriseApartment, OutPatient,
PrimarySchool, QuickServiceRestaurant, SecondarySchool, SmallHotel,
SmallOffice, Stand-aloneRetail, StripMall, SuperMarket, Warehouse.

**Target labels:** `Normal`, `Theft1`, `Theft2`, `Theft3`, `Theft4`,
`Theft5`, `Theft6`.

---

## 🧠 Modeling Pipeline

1. **Feature Engineering** (`FeatureEngineer`) — derives `block_id`,
   `hour_of_day`, `day_of_year`, `month`, `day_of_week`, `is_weekend`,
   `year_in_block` from `row_id`, adds sin/cos cyclical encodings for
   hour/day-of-week/month, and computes consumption-ratio features
   (e.g. `Fans:Electricity_Ratio`) relative to facility totals.
2. **Numeric transform** (`MixedPowerTransformer`) — per column, applies
   **Box-Cox** if all training values are strictly positive, otherwise
   **Yeo-Johnson**; falls back to an identity transform for constant
   columns. At inference, values ≤ 0 are clipped to a small epsilon before
   a Box-Cox-fitted column is transformed (Box-Cox rejects non-positive
   values even at predict time).
3. **Scaling** — `RobustScaler` on all numeric/ratio/cyclical features.
4. **Categorical encoding** — `OneHotEncoder(drop='first')` on `Class`.
5. **Model** — `XGBClassifier` (multi-class, 7 classes).

All of the above is packaged into a single `sklearn.Pipeline` named
`preprocessor`, so `preprocessor.transform(raw_df)` produces model-ready
input in one call.

---

## 🛠️ Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `This Pipeline instance is not fitted yet` | `preprocessor.joblib` was saved before `.fit()`/`.fit_transform()` was called | Re-run `preprocessor.fit_transform(X_train)` **then** `joblib.dump(...)` |
| `Can't get attribute '_RemainderColsList'` on unpickle | scikit-learn version mismatch between training and serving environments | Pin and match `scikit-learn` version exactly in both; verify with `python -c "import sklearn; print(sklearn.__version__, sklearn.__file__)"` using the **same interpreter** that runs Streamlit |
| `Feature shape mismatch, expected: N, got: M` | `preprocessor.joblib` and the model `.pkl` were saved from different runs / different `OneHotEncoder` settings (e.g. `drop='first'` present in one but not the other) | Best fix: re-fit preprocessor + model together in one run and save both. If retraining isn't possible, `app.py` auto-aligns the preprocessor's output columns to `model.get_booster().feature_names`, dropping any extras — this is already handled in the shipped app |
| `ValueError` from Box-Cox during single-row prediction | A column fit with Box-Cox received a `0` or negative value at inference | Already handled in `MixedPowerTransformer.transform` (values clipped to a small positive epsilon) |

---

## 📋 Requirements

See `requirements.txt`. Key dependencies: `streamlit`, `scikit-learn`,
`xgboost`, `pandas`, `numpy`, `plotly`, `joblib`.

---

## 📝 Notes

- The model and preprocessor **must always be re-saved together** after any
  change to either — they are only valid as a matched pair.
- `src/utils/transformers.py` and `src/utils/schema.py` are shared between
  the training notebook and `app.py`; keep them as the single source of
  truth for feature engineering and column definitions.
