import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb


# =====================================================
# Konfigurasi Halaman
# =====================================================
st.set_page_config(
    page_title="Prediksi Kualitas Air",
    page_icon="💧",
    layout="wide"
)


# =====================================================
# Helper Path Artefak
# =====================================================
BASE_DIR = Path(__file__).resolve().parent

ARTIFACT_DIR_CANDIDATES = [
    BASE_DIR / "deployment_artifacts",
    BASE_DIR,
]


def find_artifact(filename: str) -> Path:
    """Mencari file artefak pada folder app atau folder deployment_artifacts."""
    for artifact_dir in ARTIFACT_DIR_CANDIDATES:
        path = artifact_dir / filename
        if path.exists():
            return path
    raise FileNotFoundError(
        f"File '{filename}' tidak ditemukan. "
        f"Letakkan file tersebut di folder 'deployment_artifacts' atau sejajar dengan app.py."
    )


# =====================================================
# Load Artefak Deployment
# =====================================================
@st.cache_resource
def load_artifacts():
    feature_order_path = find_artifact("feature_order.json")
    imputer_path = find_artifact("feature_imputer.joblib")
    scaler_path = find_artifact("minmax_scaler.joblib")
    model_path = find_artifact("xgboost_model.json")

    with open(feature_order_path, "r", encoding="utf-8") as f:
        feature_order = json.load(f)

    imputer = joblib.load(imputer_path)
    scaler = joblib.load(scaler_path)

    model = xgb.Booster()
    model.load_model(str(model_path))

    metadata = {}
    try:
        metadata_path = find_artifact("deployment_metadata.json")
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except FileNotFoundError:
        metadata = {
            "model_name": "XGBoost + Optuna",
            "prediction_threshold": 0.5,
            "class_mapping": {
                "0": "Tidak Layak Minum",
                "1": "Layak Minum"
            }
        }

    return model, imputer, scaler, feature_order, metadata


try:
    model, imputer, scaler, feature_order, metadata = load_artifacts()
except Exception as e:
    st.error("Artefak model belum lengkap atau gagal dimuat.")
    st.exception(e)
    st.stop()


# =====================================================
# Informasi Fitur dan Nilai Default
# =====================================================
# Nilai default dapat disesuaikan dengan statistik dataset training.
DEFAULT_VALUES = {
    "ph": 7.0,
    "Hardness": 200.0,
    "Solids": 20000.0,
    "Chloramines": 7.0,
    "Sulfate": 330.0,
    "Conductivity": 425.0,
    "Organic_carbon": 14.0,
    "Trihalomethanes": 66.0,
    "Turbidity": 4.0,
}

FEATURE_DESCRIPTIONS = {
    "ph": "Tingkat keasaman air.",
    "Hardness": "Kesadahan air.",
    "Solids": "Jumlah total padatan terlarut.",
    "Chloramines": "Kadar kloramin dalam air.",
    "Sulfate": "Kadar sulfat dalam air.",
    "Conductivity": "Kemampuan air menghantarkan listrik.",
    "Organic_carbon": "Kandungan karbon organik.",
    "Trihalomethanes": "Kadar senyawa trihalometana.",
    "Turbidity": "Tingkat kekeruhan air.",
}


# =====================================================
# Fungsi Prediksi
# =====================================================
def build_input_dataframe(input_values: dict, feature_order: list[str]) -> pd.DataFrame:
    """Menyusun input user sesuai urutan fitur saat training."""
    row = {feature: input_values.get(feature, np.nan) for feature in feature_order}
    return pd.DataFrame([row], columns=feature_order)


def predict_water_quality(input_df: pd.DataFrame):
    """Melakukan preprocessing dan prediksi menggunakan artefak hasil training."""
    imputed = imputer.transform(input_df)
    scaled = scaler.transform(imputed)
    dmatrix = xgb.DMatrix(scaled, feature_names=feature_order)

    probability = float(model.predict(dmatrix)[0])
    threshold = float(metadata.get("prediction_threshold", 0.5))
    predicted_class = int(probability > threshold)

    class_mapping = metadata.get(
        "class_mapping",
        {"0": "Tidak Layak Minum", "1": "Layak Minum"}
    )
    label = class_mapping.get(str(predicted_class), str(predicted_class))

    return predicted_class, label, probability, threshold


# =====================================================
# Tampilan Utama
# =====================================================
st.title("💧 Prediksi Kualitas Air Minum")
st.caption("Aplikasi Streamlit untuk inferensi model XGBoost + Optuna.")

with st.sidebar:
    st.header("Informasi Model")
    st.write(f"**Model:** {metadata.get('model_name', 'XGBoost + Optuna')}")
    st.write(f"**Threshold:** {metadata.get('prediction_threshold', 0.5)}")
    if metadata.get("final_split"):
        st.write(f"**Final split:** {metadata.get('final_split')}")
    if metadata.get("exported_at"):
        st.write(f"**Exported at:** {metadata.get('exported_at')}")

    st.divider()
    st.write("**Artefak yang digunakan:**")
    st.write("- xgboost_model.json")
    st.write("- feature_imputer.joblib")
    st.write("- minmax_scaler.joblib")
    st.write("- feature_order.json")
    st.write("- deployment_metadata.json")

st.subheader("Input Parameter Kualitas Air")

with st.form("prediction_form"):
    input_values = {}

    cols = st.columns(3)
    for idx, feature in enumerate(feature_order):
        with cols[idx % 3]:
            input_values[feature] = st.number_input(
                label=feature,
                value=float(DEFAULT_VALUES.get(feature, 0.0)),
                step=0.01,
                format="%.4f",
                help=FEATURE_DESCRIPTIONS.get(feature, "Masukkan nilai fitur sesuai dataset training.")
            )

    submitted = st.form_submit_button("Prediksi Kualitas Air", type="primary")

if submitted:
    input_df = build_input_dataframe(input_values, feature_order)
    predicted_class, label, probability, threshold = predict_water_quality(input_df)

    st.divider()
    st.subheader("Hasil Prediksi")

    col_result, col_prob = st.columns([1, 1])

    with col_result:
        if predicted_class == 1:
            st.success(f"Prediksi: {label}")
        else:
            st.error(f"Prediksi: {label}")

    with col_prob:
        st.metric("Probabilitas Layak Minum", f"{probability:.4f}")
        st.progress(min(max(probability, 0.0), 1.0))

    with st.expander("Lihat detail input dan preprocessing"):
        st.write("**Input sesuai feature order:**")
        st.dataframe(input_df, use_container_width=True)
        st.write(f"Threshold klasifikasi: `{threshold}`")
        st.write("Aturan: probabilitas > threshold diklasifikasikan sebagai `Layak Minum`.")

    with st.expander("Catatan Interpretasi"):
        st.write(
            "Hasil prediksi ini merupakan output model machine learning berdasarkan pola pada dataset training. "
            "Hasil aplikasi tidak menggantikan pengujian laboratorium resmi untuk menentukan kelayakan air minum."
        )

else:
    st.info("Isi parameter kualitas air, lalu klik tombol prediksi.")


# =====================================================
# Upload CSV untuk Prediksi Batch
# =====================================================
st.divider()
st.subheader("Prediksi Batch dari CSV")
st.write("Upload file CSV yang memiliki kolom sesuai `feature_order.json`.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    try:
        batch_df = pd.read_csv(uploaded_file)
        missing_cols = [col for col in feature_order if col not in batch_df.columns]

        if missing_cols:
            st.error(f"Kolom berikut belum ada pada CSV: {missing_cols}")
        else:
            batch_input = batch_df[feature_order].copy()
            imputed = imputer.transform(batch_input)
            scaled = scaler.transform(imputed)
            dmatrix = xgb.DMatrix(scaled, feature_names=feature_order)

            probabilities = model.predict(dmatrix)
            threshold = float(metadata.get("prediction_threshold", 0.5))
            predictions = (probabilities > threshold).astype(int)

            class_mapping = metadata.get(
                "class_mapping",
                {"0": "Tidak Layak Minum", "1": "Layak Minum"}
            )

            result_df = batch_df.copy()
            result_df["probabilitas_layak_minum"] = probabilities
            result_df["prediksi_kelas"] = predictions
            result_df["prediksi_label"] = [class_mapping.get(str(pred), str(pred)) for pred in predictions]

            st.success("Prediksi batch berhasil diproses.")
            st.dataframe(result_df, use_container_width=True)

            csv_result = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Hasil Prediksi CSV",
                data=csv_result,
                file_name="hasil_prediksi_kualitas_air.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error("Gagal memproses file CSV.")
        st.exception(e)
