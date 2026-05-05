import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb


# =====================================================
# GT-MARU THEME (CSS Injection)
# =====================================================
def apply_gt_maru_theme():
    css = """
    <style>
    /* ===== Tokens (from Gt-maru spec) ===== */
    :root{
      --color-sky-blue:#0068ff;
      --color-sunshine-yellow:#ffff55;
      --color-bubblegum-pink:#ff8080;
      --color-tangerine:#ff9400;
      --color-lime-green:#00bf3a;
      --color-lemon-drop:#ffc800;
      --color-seafoam:#05cf9c;
      --color-slate-blue:#84bbff;
      --color-deep-space:#000000;
      --color-white-cloud:#ffffff;

      --spacing-7:7px;
      --spacing-10:10px;
      --spacing-13:13px;
      --spacing-20:20px;
      --spacing-30:30px;
      --spacing-40:40px;

      --radius-cards:30px;
      --radius-navitems:10px;

      --border-3:3px solid var(--color-deep-space);

      --text-body:16px;
      --leading-body:1.4;
      --tracking-body:0.24px;

      --text-subheading:25px;
      --leading-subheading:1.3;
      --tracking-subheading:0.38px;

      --text-heading:45px;
      --leading-heading:1;
      --tracking-heading:-1.35px;

      --text-display:187px;
      --leading-display:1;
      --tracking-display:-5.61px;

      --font-gt-maru:'GT Maru', ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
                     "Segoe UI", Roboto, sans-serif;
    }

    /* ===== Page canvas ===== */
    .stApp{
      background: var(--color-sky-blue);
      color: var(--color-deep-space);
    }

    html, body, [class*="css"]{
      font-family: var(--font-gt-maru) !important;
      letter-spacing: var(--tracking-body);
    }

    /* Layout spacing */
    .block-container{
      padding-top: 28px;
      padding-bottom: 40px;
      max-width: 1200px;
    }

    /* Sidebar: keep light surface + bold outline */
    section[data-testid="stSidebar"]{
      background: var(--color-white-cloud) !important;
      border-right: var(--border-3);
    }
    section[data-testid="stSidebar"] .block-container{
      padding-top: 24px;
    }

    /* ===== Headings ===== */
    .gt-display{
      font-size: clamp(56px, 7vw, 110px); /* Streamlit-friendly display */
      line-height: var(--leading-display);
      letter-spacing: var(--tracking-display);
      font-weight: 400;
      margin: 0;
      color: var(--color-sunshine-yellow);
      -webkit-text-stroke: 3px var(--color-deep-space);
      text-shadow: none;
    }
    .gt-title{
      font-size: var(--text-heading);
      line-height: var(--leading-heading);
      letter-spacing: var(--tracking-heading);
      font-weight: 400;
      margin: 0 0 6px 0;
      color: var(--color-sunshine-yellow);
      -webkit-text-stroke: 3px var(--color-deep-space);
    }
    .gt-caption{
      display:block;
      font-size: var(--text-body);
      line-height: var(--leading-body);
      margin-top: 10px;
      color: var(--color-deep-space);
      background: var(--color-lemon-drop);
      border: var(--border-3);
      border-radius: var(--radius-navitems);
      padding: 10px 13px;
      width: fit-content;
    }

    /* ===== Cards ===== */
    .gt-card{
      background: var(--color-white-cloud);
      border: var(--border-3);
      border-radius: var(--radius-cards);
      padding: var(--spacing-30);
    }
    .gt-card + .gt-card{
      margin-top: var(--spacing-20);
    }

    /* ===== Navigation tags ===== */
    .gt-nav{
      display:flex;
      flex-wrap:wrap;
      gap: var(--spacing-13);
      margin: var(--spacing-20) 0 var(--spacing-20) 0;
    }
    .gt-tag{
      display:inline-flex;
      align-items:center;
      justify-content:center;
      padding: var(--spacing-7) 16px;
      border-radius: var(--radius-navitems);
      border: var(--border-3);
      color: var(--color-deep-space);
      font-size: var(--text-body);
      line-height: 1.0;
      user-select:none;
      white-space: nowrap;
    }

    /* ===== Streamlit widgets: chunky outlines ===== */
    /* Inputs */
    div[data-baseweb="input"] > div,
    div[data-baseweb="select"] > div,
    div[data-baseweb="textarea"] > div{
      border: var(--border-3) !important;
      border-radius: var(--radius-navitems) !important;
      background: var(--color-white-cloud) !important;
      box-shadow: none !important;
    }

    /* Labels */
    label, .stNumberInput label, .stTextInput label, .stSelectbox label{
      color: var(--color-deep-space) !important;
      font-weight: 400 !important;
    }

    /* Buttons (no distinct CTA color → use Sunshine Yellow consistently) */
    .stButton > button{
      border: var(--border-3) !important;
      border-radius: var(--radius-navitems) !important;
      background: var(--color-sunshine-yellow) !important;
      color: var(--color-deep-space) !important;
      font-weight: 400 !important;
      padding: 10px 18px !important;
      box-shadow: none !important;
    }

    /* File uploader */
    section[data-testid="stFileUploaderDropzone"]{
      border: var(--border-3) !important;
      border-radius: var(--radius-cards) !important;
      background: var(--color-white-cloud) !important;
    }

    /* Expanders */
    details{
      border: var(--border-3) !important;
      border-radius: var(--radius-cards) !important;
      background: var(--color-white-cloud) !important;
      padding: 6px 10px;
    }

    /* Divider: bold line */
    hr{
      border: none;
      border-top: 3px solid var(--color-deep-space);
      margin: 20px 0;
    }

    /* Metrics: give them chunky card feel */
    div[data-testid="stMetric"]{
      background: var(--color-white-cloud);
      border: var(--border-3);
      border-radius: var(--radius-cards);
      padding: 18px 18px;
    }

    /* Progress bar */
    div[data-testid="stProgress"] > div{
      border: var(--border-3);
      border-radius: var(--radius-navitems);
      background: var(--color-white-cloud);
      overflow: hidden;
    }
    div[data-testid="stProgress"] div[role="progressbar"]{
      background: var(--color-seafoam) !important;
    }

    /* Dataframe container border */
    div[data-testid="stDataFrame"]{
      border: var(--border-3);
      border-radius: var(--radius-cards);
      overflow: hidden;
      background: var(--color-white-cloud);
    }

    /* Alerts: keep them cartoon-ish */
    div[data-testid="stAlert"]{
      border: var(--border-3);
      border-radius: var(--radius-cards);
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# =====================================================
# Konfigurasi Halaman
# =====================================================
st.set_page_config(
    page_title="Prediksi Kualitas Air",
    page_icon="💧",
    layout="wide",
)

apply_gt_maru_theme()


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
            "class_mapping": {"0": "Tidak Layak Minum", "1": "Layak Minum"},
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

    class_mapping = metadata.get("class_mapping", {"0": "Tidak Layak Minum", "1": "Layak Minum"})
    label = class_mapping.get(str(predicted_class), str(predicted_class))
    return predicted_class, label, probability, threshold


# =====================================================
# Tampilan Utama (Gt-maru layout)
# =====================================================
st.markdown('<h1 class="gt-display">WATER</h1>', unsafe_allow_html=True)
st.markdown('<h2 class="gt-title">Prediksi Kualitas Air Minum</h2>', unsafe_allow_html=True)
st.markdown(
    '<span class="gt-caption">Aplikasi Streamlit untuk inferensi model XGBoost + Optuna.</span>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="gt-nav">
      <span class="gt-tag" style="background:#05cf9c;">Input</span>
      <span class="gt-tag" style="background:#ff8080;">Hasil</span>
      <span class="gt-tag" style="background:#ff9400;">Batch CSV</span>
      <span class="gt-tag" style="background:#84bbff;">Model Info</span>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown('<div class="gt-card">', unsafe_allow_html=True)
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
    st.markdown("</div>", unsafe_allow_html=True)

# Section: Inputs & Results
left, right = st.columns([1.15, 0.85], gap="large")

with left:
    st.markdown('<div class="gt-card">', unsafe_allow_html=True)
    st.subheader("Input Parameter Kualitas Air")

    with st.form("prediction_form"):
        input_values = {}
        cols = st.columns(3, gap="medium")
        for idx, feature in enumerate(feature_order):
            with cols[idx % 3]:
                input_values[feature] = st.number_input(
                    label=feature,
                    value=float(DEFAULT_VALUES.get(feature, 0.0)),
                    step=0.01,
                    format="%.4f",
                    help=FEATURE_DESCRIPTIONS.get(feature, "Masukkan nilai fitur sesuai dataset training."),
                )

        submitted = st.form_submit_button("Prediksi Kualitas Air")

    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="gt-card">', unsafe_allow_html=True)
    st.subheader("Hasil Prediksi")

    if submitted:
        input_df = build_input_dataframe(input_values, feature_order)
        predicted_class, label, probability, threshold = predict_water_quality(input_df)

        if predicted_class == 1:
            st.success(f"Prediksi: {label}")
        else:
            st.error(f"Prediksi: {label}")

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

    st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# Upload CSV untuk Prediksi Batch (Card)
# =====================================================
st.markdown('<div class="gt-card">', unsafe_allow_html=True)
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

            class_mapping = metadata.get("class_mapping", {"0": "Tidak Layak Minum", "1": "Layak Minum"})

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
                mime="text/csv",
            )

    except Exception as e:
        st.error("Gagal memproses file CSV.")
        st.exception(e)

st.markdown("</div>", unsafe_allow_html=True)