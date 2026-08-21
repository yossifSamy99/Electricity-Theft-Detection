"""
Custom sklearn-compatible transformers used inside preprocessor.joblib.

⚠️ مهم جدًا:
هذا الملف لازم يفضل موجود بنفس المسار (src/utils/transformers.py) وبنفس أسماء
الكلاسات دي، لأن joblib بيحفظ الـ Pipeline بالإشارة لمسار الكلاس مش بالكود نفسه.
لو الـ preprocessor.joblib بتاعك اتعمل بكلاسات معرفة جوه النوت بوك مباشرة
(cell-level)، هتحتاج تعيد حفظه بعد ما تستورد الكلاسات من هنا، وإلا
joblib.load() هيرمي ModuleNotFoundError / AttributeError.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PowerTransformer


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """يستخرج ميزات زمنية من row_id + نسب استهلاك، ويشيل الأعمدة الخام المستخدمة."""

    BLOCK_SIZE = 8760

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        # ---- ميزات زمنية من row_id ----
        df["block_id"] = df["row_id"] // self.BLOCK_SIZE
        hour_of_year = df["row_id"] % self.BLOCK_SIZE
        df["hour_of_day"] = hour_of_year % 24
        df["day_of_year"] = hour_of_year // 24
        df["month"] = (df["day_of_year"] // 30 + 1).clip(upper=12)
        df["day_of_week"] = df["day_of_year"] % 7
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
        df["year_in_block"] = df["block_id"] % 4

        for col, period in [("hour_of_day", 24), ("day_of_week", 7), ("month", 12)]:
            df[f"{col}_sin"] = np.sin(2 * np.pi * df[col] / period)
            df[f"{col}_cos"] = np.cos(2 * np.pi * df[col] / period)

        # ---- نسب الاستهلاك ----
        df["Fans:Electricity_Ratio"] = np.where(
            df["Electricity:Facility [kW](Hourly)"] > 0,
            df["Fans:Electricity [kW](Hourly)"] / df["Electricity:Facility [kW](Hourly)"], 0)
        df["InteriorEquipment:Electricity_Ratio"] = np.where(
            df["Electricity:Facility [kW](Hourly)"] > 0,
            df["InteriorEquipment:Electricity [kW](Hourly)"] / df["Electricity:Facility [kW](Hourly)"], 0)
        df["Cooling:Electricity_Ratio"] = np.where(
            df["Electricity:Facility [kW](Hourly)"] > 0,
            df["Cooling:Electricity [kW](Hourly)"] / df["Electricity:Facility [kW](Hourly)"], 0)
        df["InteriorLights:Electricity_Ratio"] = np.where(
            df["Electricity:Facility [kW](Hourly)"] > 0,
            df["InteriorLights:Electricity [kW](Hourly)"] / df["Electricity:Facility [kW](Hourly)"], 0)
        df["Heating:Gas_Ratio"] = np.where(
            df["Gas:Facility [kW](Hourly)"] > 0,
            df["Heating:Gas [kW](Hourly)"] / df["Gas:Facility [kW](Hourly)"], 0)

        df = df.drop(columns=[
            "Fans:Electricity [kW](Hourly)", "InteriorEquipment:Electricity [kW](Hourly)",
            "InteriorLights:Electricity [kW](Hourly)", "Cooling:Electricity [kW](Hourly)",
            "Heating:Gas [kW](Hourly)", "row_id",
        ])
        return df


class _Identity:
    """بديل آمن لعمود لا يقبل أي Power Transform (مثلاً قيمة ثابتة تمامًا)."""

    def transform(self, X):
        return X

    def fit(self, X, y=None):
        return self


class MixedPowerTransformer(BaseEstimator, TransformerMixin):


    def fit(self, X, y=None):
        X = pd.DataFrame(X).copy()
        self.transformers_ = {}
        self.methods_ = {}
        for col in X.columns:
            is_positive = (X[col] > 0).all()
            method = "box-cox" if is_positive else "yeo-johnson"
            pt = PowerTransformer(method=method, standardize=False)
            try:
                pt.fit(X[[col]])
                self.transformers_[col] = pt
                self.methods_[col] = method
            except ValueError:
                
                try:
                    pt = PowerTransformer(method="yeo-johnson", standardize=False)
                    pt.fit(X[[col]])
                    self.transformers_[col] = pt
                    self.methods_[col] = "yeo-johnson (fallback)"
                except ValueError:
                    self.transformers_[col] = _Identity()
                    self.methods_[col] = "identity (constant column)"
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for col, pt in self.transformers_.items():
            col_data = X[[col]].copy()
            if self.methods_.get(col) == "box-cox":

                col_data[col] = col_data[col].clip(lower=1e-6)
            X[col] = pt.transform(col_data)
        return X

    def get_feature_names_out(self, input_features=None):
        return np.array(list(self.transformers_.keys()))
