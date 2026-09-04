# ⚡ Electricity & Gas Theft Detection

A machine learning system that classifies hourly electricity/gas consumption
readings from commercial buildings as **Normal** or one of **6 theft
patterns** (`Theft1`–`Theft6`), using an XGBoost multi-class classifier
served through a self-contained Streamlit UI — locally and on
[Streamlit Community Cloud](https://streamlit.io/cloud).

**Live demo:** `https://electricity-theft-detection-<your-id>.streamlit.app`
*(update this once your app is deployed)*

---

## 📌 Overview

Utility meter readings (electricity + gas, broken down by end-use: fans,
cooling, heating, interior lighting/equipment, water heating) are engineered
into time-cyclical and consumption-ratio features, transformed with a
per-column Box-Cox / Yeo-Johnson power transform, and fed into an XGBoost
classifier trained to distinguish normal usage from known theft/fraud
signatures.

The Streamlit app loads the trained model and preprocessing pipeline
**directly via `joblib`** — there is no separate API layer. Training happens
in a GPU-enabled notebook (Google Colab); the app itself only needs CPU.

---

## 🗂️ Project Structure

```
.
├── app.py                          # Streamlit UI (single + batch prediction)
├── final_notebook.ipynb            # EDA, feature engineering, model training
├── requirements.txt                # Pinned dependency versions (see below)
├── sample_test_batch.csv           # Ready-made CSV for testing the Batch tab
├── data/
│   └── df.csv                      # Training dataset
├── src/
│   ├── __init__.py
│   ├── artifacts/
│   │   ├── best_xgboost_model.pkl  # Trained XGBoost model
│   │   └── preprocessor.joblib     # Fitted sklearn Pipeline (feature eng. + encoding)
│   └── utils/
│       ├── __init__.py
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

## ⚙️ Local Installation

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
```

**Pin every version explicitly in `requirements.txt`** — this single file is
what keeps your local environment, your training environment (Colab), and
Streamlit Cloud all in sync:

```
streamlit==1.38.0
scikit-learn==1.6.1
xgboost==2.1.1
pandas
numpy
plotly
joblib
```

Whatever `scikit-learn`/`xgboost` versions you train with in Colab, copy
those *exact* pins here — mismatched versions are the #1 source of the
unpickling errors in the Troubleshooting section below.

---

## ▶️ Running Locally

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

Use the included `sample_test_batch.csv` to quickly test the Batch tab
end-to-end.

---

## ☁️ Deploying to Streamlit Community Cloud

1. Push the full project (including `src/`, `requirements.txt`, and the
   files in `src/artifacts/`) to a GitHub repo. If `best_xgboost_model.pkl`
   is close to or over **100 MB**, use [Git LFS](https://git-lfs.com/) —
   plain GitHub rejects files over that limit.
2. On [share.streamlit.io](https://share.streamlit.io), create a new app
   pointing to the repo, branch `main`, main file `app.py`.
3. In **Advanced settings**, set the **Python version** explicitly to
   **3.11 or 3.12**. Don't leave it on the newest available version
   (e.g. 3.14) — `scikit-learn`/`xgboost` often don't have prebuilt wheels
   for a Python version that new yet, which forces a from-source build that
   can hang for a very long time during "Processing dependencies".
4. Deploy, then watch **Manage app → Logs** live. A healthy deploy shows
   `Using Python 3.11.x` (or 3.12.x) followed by package installs, then the
   app starting — no more than a couple of minutes.
5. After any change to `requirements.txt`, model files, or code, push to
   GitHub — Streamlit Cloud rebuilds automatically. If it seems stuck,
   use **Manage app → Reboot app** to force a clean rebuild.

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

`app.py` also auto-aligns the preprocessor's output to whatever columns the
loaded model actually expects (see `_align_to_model_features` in `app.py`),
so small mismatches between a `preprocessor.joblib`/model pair from
different training runs don't crash the app outright — though re-fitting
and saving both together in one run is always the more correct fix.

---

## 🛠️ Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `This Pipeline instance is not fitted yet` | `preprocessor.joblib` was saved before `.fit()`/`.fit_transform()` was called | Re-run `preprocessor.fit_transform(X_train)` **then** `joblib.dump(...)` |
| `Can't get attribute '_RemainderColsList'` on unpickle | scikit-learn version mismatch between training and serving environments (locally, or between Colab and Streamlit Cloud) | Pin the exact same `scikit-learn` version everywhere in `requirements.txt`; verify locally with `python -c "import sklearn; print(sklearn.__version__, sklearn.__file__)"` using the **same interpreter** that runs Streamlit |
| `Feature shape mismatch, expected: N, got: M` | `preprocessor.joblib` and the model `.pkl` were saved from different runs / different `OneHotEncoder` settings (e.g. `drop='first'` present in one but not the other) | Best fix: re-fit preprocessor + model together in one run and save both. `app.py` also auto-aligns columns by name as a fallback |
| `ValueError` from Box-Cox during single-row prediction | A column fit with Box-Cox received a `0` or negative value at inference | Already handled in `MixedPowerTransformer.transform` (values clipped to a small positive epsilon) |
| `'NoneType' object is not iterable` during prediction | The model was `.fit()` on a raw `numpy` array instead of a `pandas.DataFrame`, so `model.get_booster().feature_names` is `None` | Already handled in `app.py` (falls back to matching by column **count** via `model.n_features_in_`). For a cleaner long-term fix, wrap the preprocessor's output in a `DataFrame` with column names before `model.fit(...)` |
| `ModuleNotFoundError` on Streamlit Cloud (message redacted) | Usually a missing package in `requirements.txt` (commonly `xgboost` or `plotly`), or `src/` wasn't actually pushed to GitHub | Check **Manage app → Logs** for the real module name; verify `requirements.txt` lists every import used in `app.py`; confirm `src/`, `src/utils/`, and `src/artifacts/` are present in the GitHub repo (check `.gitignore` isn't excluding them) |
| Deploy stuck for 10+ minutes on "Processing dependencies" | Streamlit Cloud picked a very new Python version (e.g. 3.14) with no prebuilt wheels yet for `scikit-learn`/`xgboost`, forcing a slow/failing from-source build | In app **Settings → Advanced settings**, set Python version to **3.11 or 3.12**, then reboot the app |

---

## 📋 Requirements

See `requirements.txt`. Key dependencies: `streamlit`, `scikit-learn`,
`xgboost`, `pandas`, `numpy`, `plotly`, `joblib`.

---

## 📝 Notes

- The model and preprocessor **must always be re-saved together**, in the
  same notebook run, after any change to either — they are only valid as a
  matched pair.
- `src/utils/transformers.py` and `src/utils/schema.py` are shared between
  the training notebook and `app.py`; keep them as the single source of
  truth for feature engineering and column definitions.
- Keep `requirements.txt` as the single source of truth for package
  versions across your local machine, Colab, and Streamlit Cloud — update
  all three together whenever you change a version.
