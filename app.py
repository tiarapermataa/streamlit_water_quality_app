import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb


# =====================================================
# UI THEME (LIGHT MODE ONLY)
# =====================================================
def apply_vercel_theme():
    css = """
    <style>

    :root{
        --bg: #ffffff;
        --surface: #ffffff;
        --text: #171717;
        --muted: #4d4d4d;
        --line: rgba(0,0,0,0.08);
        --line-soft: rgba(0,0,0,0.06);
        --shadow-soft: rgba(0,0,0,0.04);
        --blue: #0a72ef;
        --pink: #de1d8d;
        --red: #ff5b4f;
        --badge-bg: #ebf5ff;
        --badge-text: #0068d6;
        --focus: hsla(212, 100%, 48%, 1);
        --radius-pill: 9999px;
        --max: 1200px;
    }

    .stApp{
        background:
            radial-gradient(circle at top left, rgba(10,114,239,0.06), transparent 28%),
            radial-gradient(circle at top right, rgba(222,29,141,0.05), transparent 22%),
            linear-gradient(180deg, var(--bg) 0%, var(--bg) 100%);
        color: var(--text);
    }

    html, body, [class*="css"]{
        font-family: Arial, sans-serif !important;
        color: var(--text);
    }

    .block-container{
        max-width: var(--max);
        padding-top: 28px;
        padding-bottom: 64px;
    }

    .hero{
        padding: 10px 0 20px 0;
        margin-bottom: 18px;
    }

    .eyebrow{
        display: inline-flex;
        padding: 6px 12px;
        border-radius: var(--radius-pill);
        color: var(--badge-text);
        background: var(--badge-bg);
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
    }

    .hero h1{
        margin: 14px 0 8px 0;
        font-size: clamp(42px, 6vw, 64px);
        line-height: .98;
        letter-spacing: -2.4px;
        font-weight: 600;
    }

    .hero p{
        margin: 0;
        max-width: 760px;
        font-size: 18px;
        line-height: 1.65;
        color: var(--muted);
    }

    .panel{
        background: var(--surface);
        border-radius: 18px;
        box-shadow: 0 0 0 1px var(--line);
        padding: 22px;
    }

    .section-lead{
        color: var(--muted);
        line-height: 1.6;
        margin: 0 0 18px 0;
    }

    .summary-card{
        border-radius: 16px;
        padding: 16px;
        background: var(--surface);
        box-shadow: 0 0 0 1px var(--line);
    }

    .summary-card strong{
        display:block;
        font-size: 13px;
        text-transform: uppercase;
        color: var(--muted);
        margin-bottom: 8px;
    }

    .summary-card span{
        font-size: 15px;
        line-height: 1.5;
        color: var(--text);
    }

    .stButton > button,
    .stDownloadButton > button{
        border-radius: 12px !important;
        padding: 10px 18px !important;
        font-weight: 600 !important;
        border: none !important;
        background: var(--text) !important;
        color: white !important;
    }

    div[data-testid="stMetric"]{
        background: var(--surface);
        box-shadow: 0 0 0 1px var(--line);
        border-radius: 14px;
        padding: 16px;
    }

    div[data-testid="stDataFrame"]{
        box-shadow: 0 0 0 1px var(--line);
        border-radius: 14px;
        overflow: hidden;
    }

    @media (max-width: 900px){
        .hero h1{
            font-size: clamp(34px, 10vw, 48px);
        }
    }

    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Prediksi Kualitas Air",
    page_icon="💧",
    layout="wide",
)

apply_vercel_theme()


# =====================================================
# PATH CONFIG
# =====================================================
BASE_DIR = Path(__file__).resolve().parent

ARTIFACT_DIR_CANDIDATES = [
    BASE_DIR / "deployment_artifacts",
    BASE_DIR,
]


def find_artifact(filename: str) -> Path:
    for artifact_dir in ARTIFACT_DIR_CANDIDATES:
        path = artifact_dir / filename
        if path.exists():
            return path

    raise FileNotFoundError(
        f"File '{filename}' tidak ditemukan."
    )


# =====================================================
# LOAD ARTIFACTS
# =====================================================
@st.cache_resource
def load_shared_artifacts():
    feature_order_path = find_artifact("feature_order.json")
    imputer_path = find_artifact("feature_imputer.joblib")
    scaler_path = find_artifact("minmax_scaler.joblib")

    with open(feature_order_path, "r", encoding="utf-8") as f:
        feature_order = json.load(f)

    imputer = joblib.load(imputer_path)
    scaler = joblib.load(scaler_path)

    return imputer, scaler, feature_order


@st.cache_resource
def load_model_registry():
    registry_path = find_artifact("model_registry.json")

    with open(registry_path, "r", encoding="utf-8") as f:
        registry = json.load(f)

    return registry


@st.cache_resource
def load_xgb_model(model_json_path: str):
    model = xgb.Booster()
    model.load_model(str(model_json_path))
    return model


@st.cache_data
def load_metrics_registry():
    metrics_path = find_artifact("model_metrics_registry.csv")
    return pd.read_csv(metrics_path)


try:
    imputer, scaler, feature_order = load_shared_artifacts()
    registry = load_model_registry()
    metrics_df = load_metrics_registry()

except Exception as e:
    st.error("Artefak gagal dimuat.")
    st.exception(e)
    st.stop()


# =====================================================
# DEFAULT VALUES
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
    "Trihalomethanes": "Kadar trihalometana.",
    "Turbidity": "Tingkat kekeruhan air.",
}


# =====================================================
# HELPER FUNCTIONS
# =====================================================
def build_input_df(input_values, feature_order):
    row = {
        feature: input_values.get(feature, np.nan)
        for feature in feature_order
    }

    return pd.DataFrame([row], columns=feature_order)


def predict_xgb_booster(model, dmatrix, best_iteration=None):

    if best_iteration is not None and best_iteration >= 0:
        try:
            return model.predict(
                dmatrix,
                iteration_range=(0, int(best_iteration) + 1)
            )

        except Exception:
            pass

    return model.predict(dmatrix)


def predict_quality(input_df, model, best_iteration=None):

    imputed = imputer.transform(input_df)
    scaled = scaler.transform(imputed)

    dmatrix = xgb.DMatrix(
        scaled,
        feature_names=feature_order
    )

    probability = float(
        predict_xgb_booster(
            model,
            dmatrix,
            best_iteration
        )[0]
    )

    threshold = 0.5

    predicted_class = int(probability > threshold)

    class_mapping = {
        "0": "Tidak Layak Minum",
        "1": "Layak Minum"
    }

    label = class_mapping.get(
        str(predicted_class),
        str(predicted_class)
    )

    return predicted_class, label, probability


# =====================================================
# SIDEBAR
# =====================================================
with st.sidebar:

    st.markdown("## Navigasi")

    page = st.radio(
        "Pilih Halaman",
        ["Home", "Prediksi"]
    )

    st.divider()

    st.markdown("### 📊 Pilih Split Data")

    available_splits = sorted(
        metrics_df["split"].unique().tolist()
    )

    selected_split = st.selectbox(
        "Split Data",
        available_splits
    )


# =====================================================
# LOAD MODELS
# =====================================================
baseline_model_info = None
optuna_model_info = None

for model_key, model_info in registry["models"].items():

    model_type = model_info.get("model_type", "").lower()
    split = model_info.get("split")

    if split == selected_split:

        if "optuna" in model_type:
            optuna_model_info = model_info

        else:
            baseline_model_info = model_info


if optuna_model_info is None:
    st.error("Model Optuna tidak ditemukan.")
    st.stop()

model_json_path = find_artifact(
    optuna_model_info["model_json"]
)

selected_model = load_xgb_model(model_json_path)

best_iteration = optuna_model_info.get(
    "best_iteration"
)


# =====================================================
# HOME PAGE
# =====================================================
def render_home():
    st.markdown(
        """
        <section class="hero">
            <span class="eyebrow">Water Potability Prediction</span>

            <h1>Prediksi Kualitas Air Minum</h1>

            <p>
            Sistem prediksi kualitas air minum menggunakan
            algoritma XGBoost dan optimasi hyperparameter Optuna
            berdasarkan parameter kimia air.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    col1.metric(
        "Jumlah Model",
        len(registry["models"])
    )

    col2.metric(
        "Jumlah Fitur",
        len(feature_order)
    )

    st.markdown("### Parameter Input")

    feature_df = pd.DataFrame({
        "Fitur": list(FEATURE_DESCRIPTIONS.keys()),
        "Deskripsi": list(FEATURE_DESCRIPTIONS.values())
    })

    st.dataframe(
        feature_df,
        use_container_width=True
    )

    st.markdown("### Kelas Output")

    st.write("0 → Tidak Layak Minum")
    st.write("1 → Layak Minum")


# =====================================================
# MODEL COMPARISON
# =====================================================
def render_model_comparison(selected_split):

    split_df = metrics_df[
        metrics_df["split"] == selected_split
    ]

    baseline_df = split_df[
        split_df["model_type"]
        .str.contains("baseline", case=False)
    ]

    optuna_df = split_df[
        split_df["model_type"]
        .str.contains("optuna", case=False)
    ]

    if baseline_df.empty or optuna_df.empty:
        st.warning(
            "Perbandingan baseline dan Optuna belum tersedia."
        )
        return

    baseline = baseline_df.iloc[0]
    optuna = optuna_df.iloc[0]

    comparison_df = pd.DataFrame({
        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1-Score"
        ],

        "XGBoost Baseline": [
            baseline["accuracy"],
            baseline["precision"],
            baseline["recall"],
            baseline["f1_score"]
        ],

        "XGBoost + Optuna": [
            optuna["accuracy"],
            optuna["precision"],
            optuna["recall"],
            optuna["f1_score"]
        ]
    })

    st.markdown("## Perbandingan Performa Model")

    chart_df = comparison_df.set_index("Metric")

    st.bar_chart(chart_df)

    st.dataframe(
        comparison_df,
        use_container_width=True
    )

    improvement = (
        (
            optuna["f1_score"]
            - baseline["f1_score"]
        )
        / baseline["f1_score"]
    ) * 100

    st.metric(
        "Peningkatan F1-Score Optuna",
        f"{improvement:.2f}%"
    )

    st.success(
        "Model XGBoost + Optuna memberikan performa terbaik berdasarkan F1-Score."
    )

    st.markdown("## Confusion Matrix")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### XGBoost Baseline")

        baseline_cm = baseline.get(
            "confusion_matrix",
            {}
        )

        baseline_cm_df = pd.DataFrame(
            [
                [
                    baseline_cm.get("tn", 0),
                    baseline_cm.get("fp", 0)
                ],
                [
                    baseline_cm.get("fn", 0),
                    baseline_cm.get("tp", 0)
                ]
            ],

            index=[
                "Actual Tidak Layak",
                "Actual Layak"
            ],

            columns=[
                "Pred Tidak Layak",
                "Pred Layak"
            ]
        )

        st.dataframe(
            baseline_cm_df,
            use_container_width=True
        )

    with col2:

        st.markdown("### XGBoost + Optuna")

        optuna_cm = optuna.get(
            "confusion_matrix",
            {}
        )

        optuna_cm_df = pd.DataFrame(
            [
                [
                    optuna_cm.get("tn", 0),
                    optuna_cm.get("fp", 0)
                ],
                [
                    optuna_cm.get("fn", 0),
                    optuna_cm.get("tp", 0)
                ]
            ],

            index=[
                "Actual Tidak Layak",
                "Actual Layak"
            ],

            columns=[
                "Pred Tidak Layak",
                "Pred Layak"
            ]
        )

        st.dataframe(
            optuna_cm_df,
            use_container_width=True
        )


# =====================================================
# PREDICTION PAGE
# =====================================================
def render_prediction():

    st.markdown(
        """
        <section class="hero">

            <span class="eyebrow">
            Prediksi Kualitas Air
            </span>

            <h1>
            Masukkan Parameter Air
            </h1>

            <p>
            Isi nilai parameter sampel air untuk
            mendapatkan hasil prediksi kualitas air
            serta perbandingan performa model.
            </p>

        </section>
        """,
        unsafe_allow_html=True,
    )

    input_col, result_col = st.columns(
        [1.2, 0.8],
        gap="large"
    )

    submitted = False

    with input_col:

        st.subheader("Input Parameter")

        with st.form("prediction_form"):

            input_values = {}

            cols = st.columns(3)

            for idx, feature in enumerate(feature_order):

                with cols[idx % 3]:

                    input_values[feature] = st.number_input(
                        label=feature,

                        value=float(
                            DEFAULT_VALUES.get(
                                feature,
                                0.0
                            )
                        ),

                        step=0.01,

                        format="%.4f",

                        help=FEATURE_DESCRIPTIONS.get(
                            feature,
                            ""
                        ),
                    )

            submitted = st.form_submit_button(
                "Prediksi Sekarang"
            )

    with result_col:

        st.subheader("Hasil Prediksi")

        if submitted:

            input_df = build_input_df(
                input_values,
                feature_order
            )

            predicted_class, label, probability = (
                predict_quality(
                    input_df,
                    selected_model,
                    best_iteration
                )
            )

            if predicted_class == 1:
                st.success(f"Prediksi: {label}")

            else:
                st.error(f"Prediksi: {label}")

            st.metric(
                "Probabilitas Layak Minum",
                f"{probability:.4f}"
            )

            st.progress(
                min(max(probability, 0.0), 1.0)
            )

            st.markdown(
                f"""
                <div class="summary-card">
                    <strong>Hasil Klasifikasi</strong>
                    <span>{label}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        else:

            st.info(
                "Masukkan parameter lalu klik Prediksi Sekarang."
            )

    if submitted:

        st.divider()

        render_model_comparison(
            selected_split
        )


# =====================================================
# ROUTING
# =====================================================
if page == "Home":
    render_home()

elif page == "Prediksi":
    render_prediction()