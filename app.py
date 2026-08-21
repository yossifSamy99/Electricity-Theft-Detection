"""
Electricity Theft Detection — Streamlit App
=============================================
يحمّل best_xgboost_model.pkl + preprocessor.joblib مباشرة (من غير API)
ويقدم تبويبين: توقع فردي (Single) وتوقع دفعة كاملة (Batch CSV).

طريقة التشغيل:
    streamlit run app.py
"""

import datetime as dt
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# لازم يتستوردوا قبل joblib.load عشان الـ Pipeline يعرف يفك الكلاسات المخصصة
from src.utils.transformers import FeatureEngineer, MixedPowerTransformer  # noqa: F401
from src.utils.schema import (
    RAW_NUMERIC_COLUMNS,
    BUILDING_CLASSES,
    THEFT_LABELS,
    REQUIRED_BATCH_COLUMNS,
)

# ------------------------------------------------------------------ #
# إعداد الصفحة + تنسيق عام
# ------------------------------------------------------------------ #
st.set_page_config(
    page_title="Electricity Theft Detection",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .main > div { padding-top: 1.5rem; }
        .stTabs [data-baseweb="tab-list"] { gap: 8px; }
        .stTabs [data-baseweb="tab"] {
            padding: 10px 20px; border-radius: 8px 8px 0 0; font-weight: 600;
        }
        .metric-card {
            background: #ffffff10; border: 1px solid #ffffff22;
            border-radius: 12px; padding: 18px 22px; text-align: center;
        }
        .result-normal {
            background: linear-gradient(135deg,#0f5132,#146c43);
            color:white; padding:22px; border-radius:14px; text-align:center;
        }
        .result-theft {
            background: linear-gradient(135deg,#842029,#b02a37);
            color:white; padding:22px; border-radius:14px; text-align:center;
        }
        .result-label { font-size: 1.6rem; font-weight: 800; margin: 0; }
        .result-sub { opacity:.85; font-size:.9rem; margin-top:4px; }
        footer {visibility:hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

ARTIFACTS_DIR = Path(__file__).parent / "src" / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "best_xgboost_model.pkl"
PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocessor.joblib"
LABEL_ENCODER_PATH = ARTIFACTS_DIR / "label_encoder.joblib"  # اختياري


# ------------------------------------------------------------------ #
# تحميل الموديل والـ preprocessor (مرة واحدة، مع كاش)
# ------------------------------------------------------------------ #
@st.cache_resource(show_spinner="جاري تحميل الموديل والـ preprocessor...")
def load_artifacts():
    if not MODEL_PATH.exists() or not PREPROCESSOR_PATH.exists():
        return None, None, None
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    label_encoder = None
    if LABEL_ENCODER_PATH.exists():
        label_encoder = joblib.load(LABEL_ENCODER_PATH)
    return model, preprocessor, label_encoder


model, preprocessor, label_encoder = load_artifacts()
class_labels = list(label_encoder.classes_) if label_encoder is not None else THEFT_LABELS


def _clean_feature_name(name: str) -> str:
    """يشيل بادئة ColumnTransformer (num__/cat__/remainder__) وينضف الأقواس زي ما الموديل اتدرب."""
    name = re.sub(r"^(num__|cat__|remainder__)", "", name)
    name = re.sub(r"[\[\]<]", "_", name)
    return name


def _align_to_model_features(X_processed, df_raw_index) -> pd.DataFrame:
    """
    يبني DataFrame بأسماء الأعمدة الناتجة من الـ preprocessor، وبعدين يحاذيها
    تلقائيًا (يختار نفس الأعمدة بنفس الترتيب) مع الأعمدة اللي الموديل فعليًا
    اتدرب عليها — عشان أي اختلاف بين preprocessor.joblib والموديل (زي عمود
    Class إضافي بسبب drop='first' مش متطابقة) يتصلح تلقائيًا من غير ما نلمس
    الموديل أو نعيد تدريبه.
    """
    ct = preprocessor.named_steps["column_transformer"]
    raw_names = ct.get_feature_names_out()
    clean_names = [_clean_feature_name(n) for n in raw_names]
    X_df = pd.DataFrame(X_processed, columns=clean_names, index=df_raw_index)

    model_features = list(model.get_booster().feature_names)
    missing = [c for c in model_features if c not in X_df.columns]
    if missing:
        raise ValueError(
            "أعمدة مطلوبة من الموديل مش موجودة في ناتج الـ preprocessor: "
            + ", ".join(missing)
        )

    extra = [c for c in X_df.columns if c not in model_features]
    if extra:
        # عمود/أعمدة زيادة (زي فئة Class مش متوقعة من الموديل) - بتتشال تلقائيًا
        st.session_state["_last_dropped_features"] = extra

    return X_df[model_features]  # نفس الأعمدة وبنفس الترتيب اللي الموديل يعرفه


def predict(df_raw: pd.DataFrame) -> pd.DataFrame:
    """يشغل الـ pipeline كامل (preprocessor -> محاذاة الأعمدة -> model) على DataFrame خام."""
    X_processed = preprocessor.transform(df_raw)
    X_aligned = _align_to_model_features(X_processed, df_raw.index)
    proba = model.predict_proba(X_aligned)
    pred_idx = proba.argmax(axis=1)
    result = df_raw.copy()
    result["predicted_class"] = [class_labels[i] for i in pred_idx]
    result["confidence"] = proba.max(axis=1)
    return result, proba


# ------------------------------------------------------------------ #
# Sidebar
# ------------------------------------------------------------------ #
with st.sidebar:
    st.markdown("## ⚡ Electricity Theft Detection")
    st.caption("كشف سرقة الكهرباء والغاز باستخدام XGBoost")
    st.divider()

    status_ok = model is not None and preprocessor is not None
    if status_ok:
        st.success("✅ الموديل والـ preprocessor محمّلين")
    else:
        st.error("⚠️ مش لاقي الملفات في src/artifacts/")
        st.code(
            f"المطلوب:\n{MODEL_PATH.name}\n{PREPROCESSOR_PATH.name}",
            language="text",
        )

    st.divider()
    st.markdown("### 📌 عن المشروع")
    st.markdown(
        "- **الموديل:** XGBoost (multi-class)\n"
        "- **الفئات:** Normal + 6 أنواع سرقة\n"
        "- **الـ Feature Engineering:** ميزات زمنية دورية + نسب استهلاك\n"
        "- **الـ Encoding:** OneHotEncoder + Box-Cox/Yeo-Johnson + RobustScaler"
    )
    dropped = st.session_state.get("_last_dropped_features")
    if dropped:
        st.divider()
        st.caption("⚠️ أعمدة اتشالت تلقائيًا عشان تتوافق مع الموديل:")
        st.code(", ".join(dropped), language="text")

    st.divider()
    st.caption("Built with Streamlit • Model artifacts loaded locally via joblib")


# ------------------------------------------------------------------ #
# Header
# ------------------------------------------------------------------ #
st.title("⚡ نظام كشف سرقة الكهرباء والغاز")
st.caption("توقع ما إذا كانت قراءة استهلاك معينة طبيعية أو تدل على نوع من أنواع السرقة")

if not status_ok:
    st.warning(
        "الرجاء وضع `best_xgboost_model.pkl` و `preprocessor.joblib` داخل "
        "`src/artifacts/` ثم إعادة تشغيل التطبيق."
    )
    st.stop()

tab_single, tab_batch = st.tabs(["🔍 توقع فردي (Single)", "📁 توقع دفعة كاملة (Batch CSV)"])

# ==================================================================== #
# TAB 1 — Single Prediction
# ==================================================================== #
with tab_single:
    st.subheader("أدخل قراءة الاستهلاك يدويًا")

    with st.form("single_prediction_form"):
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("**⚡ قراءات الكهرباء (kW)**")
            electricity_facility = st.number_input("Electricity:Facility", min_value=0.0, value=15.0, step=0.1)
            fans_electricity = st.number_input("Fans:Electricity", min_value=0.0, value=1.0, step=0.1)
            cooling_electricity = st.number_input("Cooling:Electricity", min_value=0.0, value=0.0, step=0.1)
            heating_electricity = st.number_input("Heating:Electricity", min_value=0.0, value=0.0, step=0.1)
            interior_lights = st.number_input("InteriorLights:Electricity", min_value=0.0, value=1.5, step=0.1)
            interior_equipment_elec = st.number_input("InteriorEquipment:Electricity", min_value=0.0, value=7.5, step=0.1)

        with col_right:
            st.markdown("**🔥 قراءات الغاز (kW)**")
            gas_facility = st.number_input("Gas:Facility", min_value=0.0, value=3.5, step=0.1)
            heating_gas = st.number_input("Heating:Gas", min_value=0.0, value=0.0, step=0.1)
            interior_equipment_gas = st.number_input("InteriorEquipment:Gas", min_value=0.0, value=3.3, step=0.1)
            water_heater_gas = st.number_input("Water Heater:WaterSystems:Gas", min_value=0.0, value=0.5, step=0.1)

            st.markdown("**🏢 بيانات إضافية**")
            building_class = st.selectbox("نوع المبنى (Class)", BUILDING_CLASSES)
            reading_date = st.date_input("تاريخ القراءة", value=dt.date.today())
            reading_hour = st.slider("ساعة القراءة (0-23)", 0, 23, 12)

        submitted = st.form_submit_button("🔎 توقع الآن", use_container_width=True, type="primary")

    if submitted:
        # بناء row_id متوافق مع نفس منطق FeatureEngineer (block_id=0)
        day_of_year = reading_date.timetuple().tm_yday
        synthetic_row_id = day_of_year * 24 + reading_hour

        input_row = pd.DataFrame([{
            "row_id": synthetic_row_id,
            "Electricity:Facility [kW](Hourly)": electricity_facility,
            "Fans:Electricity [kW](Hourly)": fans_electricity,
            "Cooling:Electricity [kW](Hourly)": cooling_electricity,
            "Heating:Electricity [kW](Hourly)": heating_electricity,
            "InteriorLights:Electricity [kW](Hourly)": interior_lights,
            "InteriorEquipment:Electricity [kW](Hourly)": interior_equipment_elec,
            "Gas:Facility [kW](Hourly)": gas_facility,
            "Heating:Gas [kW](Hourly)": heating_gas,
            "InteriorEquipment:Gas [kW](Hourly)": interior_equipment_gas,
            "Water Heater:WaterSystems:Gas [kW](Hourly)": water_heater_gas,
            "Class": building_class,
        }])

        try:
            result, proba = predict(input_row)
            pred_label = result["predicted_class"].iloc[0]
            confidence = result["confidence"].iloc[0]

            st.divider()
            res_col, chart_col = st.columns([1, 1.4])

            with res_col:
                css_class = "result-normal" if pred_label == "Normal" else "result-theft"
                icon = "✅" if pred_label == "Normal" else "🚨"
                st.markdown(
                    f"""
                    <div class="{css_class}">
                        <p class="result-label">{icon} {pred_label}</p>
                        <p class="result-sub">نسبة الثقة: {confidence:.1%}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.metric("مستوى الثقة", f"{confidence:.1%}")

            with chart_col:
                proba_df = pd.DataFrame({"الفئة": class_labels, "الاحتمالية": proba[0]}).sort_values(
                    "الاحتمالية", ascending=True
                )
                fig = px.bar(
                    proba_df, x="الاحتمالية", y="الفئة", orientation="h",
                    text_auto=".1%", title="توزيع الاحتمالية عبر كل الفئات",
                )
                fig.update_layout(height=320, margin=dict(l=0, r=10, t=40, b=0))
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"حصل خطأ أثناء التوقع: {e}")


# ==================================================================== #
# TAB 2 — Batch Prediction (CSV)
# ==================================================================== #
with tab_batch:
    st.subheader("ارفع ملف CSV لتوقع مجموعة قراءات دفعة واحدة")
    st.caption(
        "الملف لازم يحتوي على الأعمدة: " + ", ".join(f"`{c}`" for c in REQUIRED_BATCH_COLUMNS)
    )

    uploaded_file = st.file_uploader("اختر ملف CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            df_batch = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"تعذر قراءة الملف: {e}")
            st.stop()

        missing_cols = [c for c in REQUIRED_BATCH_COLUMNS if c not in df_batch.columns]
        if missing_cols:
            st.error("الأعمدة الناقصة في الملف: " + ", ".join(f"`{c}`" for c in missing_cols))
            st.stop()

        st.success(f"تم رفع {len(df_batch):,} صف بنجاح")

        with st.expander("👀 معاينة أول 10 صفوف من الملف الخام"):
            st.dataframe(df_batch.head(10), use_container_width=True)

        if st.button("🚀 شغّل التوقع على الملف كامل", type="primary"):
            with st.spinner("جاري تشغيل الـ pipeline والتوقع..."):
                try:
                    result_df, proba_matrix = predict(df_batch)
                except Exception as e:
                    st.error(f"حصل خطأ أثناء التوقع: {e}")
                    st.stop()

            st.divider()
            st.subheader("📊 ملخص النتائج")

            m1, m2, m3, m4 = st.columns(4)
            normal_count = (result_df["predicted_class"] == "Normal").sum()
            theft_count = len(result_df) - normal_count
            m1.metric("إجمالي الصفوف", f"{len(result_df):,}")
            m2.metric("طبيعي (Normal)", f"{normal_count:,}")
            m3.metric("سرقة محتملة", f"{theft_count:,}", delta=f"{theft_count/len(result_df):.1%}")
            m4.metric("متوسط الثقة", f"{result_df['confidence'].mean():.1%}")

            dist_col, table_col = st.columns([1, 1.5])
            with dist_col:
                dist_df = result_df["predicted_class"].value_counts().reset_index()
                dist_df.columns = ["الفئة", "العدد"]
                fig = px.pie(dist_df, names="الفئة", values="العدد", title="توزيع التوقعات", hole=0.45)
                fig.update_layout(height=380)
                st.plotly_chart(fig, use_container_width=True)

            with table_col:
                st.markdown("**النتائج التفصيلية**")
                display_cols = ["row_id", "Class", "predicted_class", "confidence"]
                styled = result_df[display_cols].copy()
                styled["confidence"] = styled["confidence"].map(lambda x: f"{x:.1%}")
                st.dataframe(styled, use_container_width=True, height=380)

            csv_bytes = result_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "⬇️ تحميل النتائج كاملة (CSV)",
                data=csv_bytes,
                file_name="theft_predictions.csv",
                mime="text/csv",
                use_container_width=True,
            )
    else:
        st.info("لسه مفيش ملف مرفوع. ارفع CSV بنفس شكل بيانات التدريب عشان تبدأ.")
