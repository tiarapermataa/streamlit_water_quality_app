import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb


# =====================================================
# New UI System (from scratch)
# =====================================================
def apply_vercel_theme():
        css = """
        <style>
        /* Default = light */
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
            --radius-sm: 6px;
            --radius-md: 8px;
            --radius-lg: 14px;
            --radius-pill: 9999px;
            --max: 1200px;
        }

        /* Streamlit theme detection */
        html[data-theme="dark"], 
        html[data-theme="dark"] :root {
            --bg: #0f0f0f;
            --surface: #1a1a1a;
            --text: #ececec;
            --muted: #b0b0b0;
            --line: rgba(255,255,255,0.08);
            --line-soft: rgba(255,255,255,0.06);
            --shadow-soft: rgba(0,0,0,0.3);
            --badge-bg: #1e3a5f;
            --badge-text: #6da8ff;
        }

        /* Fallback jika browser pakai dark mode tapi Streamlit tidak set data-theme */
        @media (prefers-color-scheme: dark){
            :root{
                --bg: #0f0f0f;
                --surface: #1a1a1a;
                --text: #ececec;
                --muted: #b0b0b0;
                --line: rgba(255,255,255,0.08);
                --line-soft: rgba(255,255,255,0.06);
                --shadow-soft: rgba(0,0,0,0.3);
                --badge-bg: #1e3a5f;
                --badge-text: #6da8ff;
            }
        }

        .stApp{
            background:
                radial-gradient(circle at top left, rgba(10,114,239,0.06), transparent 28%),
                radial-gradient(circle at top right, rgba(222,29,141,0.05), transparent 22%),
                linear-gradient(180deg, var(--bg) 0%, var(--bg) 100%);
            color: var(--text);
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
        }

        html, body, [class*="css"]{
            font-family: Geist, Arial, "Segoe UI", Roboto, "Helvetica Neue", sans-serif !important;
            color: var(--text);
        }

        .block-container{
            max-width: var(--max);
            padding-top: 28px;
            padding-bottom: 64px;
        }

        /* Keep Streamlit toolbar and controls visible */
        .stApp [data-testid="stDecoration"]{
            display:none !important;
        }
        
        .stApp header {
            background: var(--surface);
            border-bottom: 1px solid var(--line);
            visibility: visible !important;
            display: flex !important;
        }
        
        .stApp [data-testid="stToolbar"] {
            visibility: visible !important;
            display: flex !important;
        }
        
        .stApp [data-testid="stToolbar"] > div {
            visibility: visible !important;
            display: flex !important;
        }
        
        .stApp [data-testid="stToolbar"] button {
            visibility: visible !important;
            display: inline-flex !important;
            opacity: 1 !important;
        }

        section[data-testid="stSidebar"]{
            background: var(--surface) !important;
            box-shadow: 0 0 0 1px var(--line);
            visibility: visible !important;
            display: block !important;
        }
        
        section[data-testid="stSidebar"] > div {
            visibility: visible !important;
        }

        .hero{
            padding: 10px 0 20px 0;
            margin-bottom: 18px;
            position: relative;
        }

        .eyebrow{
            display: inline-flex;
            gap: 8px;
            align-items: center;
            padding: 6px 12px;
            border-radius: var(--radius-pill);
            box-shadow: 0 0 0 1px var(--line);
            color: var(--badge-text);
            background: var(--badge-bg);
            font-size: 12px;
            font-weight: 600;
            letter-spacing: .08em;
            text-transform: uppercase;
        }

        .hero h1{
            margin: 14px 0 8px 0;
            font-size: clamp(42px, 6vw, 64px);
            line-height: .98;
            letter-spacing: -2.4px;
            font-weight: 600;
            color: var(--text);
        }

        .hero p{
            margin: 0;
            max-width: 760px;
            font-size: 18px;
            line-height: 1.65;
            color: var(--muted);
        }

        .hero-row{
            display: grid;
            grid-template-columns: 1.35fr .85fr;
            gap: 16px;
            align-items: stretch;
            margin-top: 22px;
        }

        .hero-card{
            padding: 20px;
            border-radius: 18px;
            background: rgba(255,255,255,0.88);
            box-shadow: 0 0 0 1px var(--line), 0 2px 2px var(--shadow-soft);
        }

        .hero-card-title{
            font-size: 14px;
            font-weight: 600;
            color: var(--muted);
            text-transform: uppercase;
            letter-spacing: .06em;
            margin-bottom: 12px;
        }

        .stack-badges{
            display:flex;
            flex-wrap:wrap;
            gap:10px;
        }

        .workflow{
            display:grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 12px;
        }

        .workflow-step{
            border-radius: 16px;
            background: var(--surface);
            box-shadow: 0 0 0 1px var(--line), 0 2px 2px var(--shadow-soft);
            padding: 16px;
            min-height: 126px;
        }

        .workflow-step .kicker{
            font-size: 12px;
            font-weight: 700;
            letter-spacing: .08em;
            text-transform: uppercase;
            margin-bottom: 10px;
        }

        .workflow-step h3{
            margin: 0 0 8px 0;
            font-size: 24px;
            line-height: 1.08;
            letter-spacing: -1px;
            font-weight: 600;
        }

        .workflow-step p{
            margin: 0;
            color: var(--muted);
            line-height: 1.55;
            font-size: 14px;
        }

        .blue{ color: var(--blue); }
        .pink{ color: var(--pink); }
        .red{ color: var(--red); }

        .content-grid{
            display:grid;
            grid-template-columns: 1.08fr .92fr;
            gap: 18px;
            margin-top: 18px;
        }

        .panel{
            background: var(--surface);
            border-radius: 18px;
            box-shadow: 0 0 0 1px var(--line), 0 2px 2px var(--shadow-soft);
            padding: 22px;
        }

        .panel h2,
        .panel h3{
            margin-top: 0;
            margin-bottom: 10px;
            letter-spacing: -1px;
            color: var(--text);
        }

        .section-lead{
            color: var(--muted);
            line-height: 1.6;
            margin: 0 0 18px 0;
        }

        .metric-strip{
            display:grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 12px;
            margin: 18px 0 0 0;
        }

        .metric-card{
            background: var(--surface);
            border-radius: 14px;
            box-shadow: 0 0 0 1px var(--line);
            padding: 16px;
        }

        .metric-card .label{
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: .08em;
            color: var(--muted);
            margin-bottom: 8px;
            font-weight: 700;
        }

        .metric-card .value{
            font-size: 28px;
            line-height: 1;
            letter-spacing: -1.2px;
            font-weight: 600;
            color: var(--text);
            margin-bottom: 6px;
        }

        .metric-card .note{
            color: var(--muted);
            font-size: 13px;
            line-height: 1.45;
        }

        .hero-summary{
            margin-top: 18px;
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
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
            letter-spacing: .08em;
            color: var(--muted);
            margin-bottom: 8px;
        }

        .summary-card span{
            font-size: 15px;
            line-height: 1.5;
            color: var(--text);
        }

        .stTabs [data-baseweb="tab-list"]{
            gap: 8px;
            background: transparent;
        }

        .stTabs [data-baseweb="tab"]{
            border-radius: 9999px;
            padding: 8px 14px;
            box-shadow: 0 0 0 1px var(--line);
            font-weight: 600;
            color: var(--text);
            background: var(--surface);
        }

        .stTabs [aria-selected="true"]{
            background: var(--text) !important;
            color: var(--surface) !important;
            box-shadow: none !important;
        }

        label, .stNumberInput label, .stTextInput label, .stSelectbox label{
            color: var(--text) !important;
            font-weight: 600 !important;
        }

        div[data-baseweb="input"] > div,
        div[data-baseweb="select"] > div,
        div[data-baseweb="textarea"] > div{
            box-shadow: 0 0 0 1px var(--line) !important;
            border-radius: 12px !important;
            background: var(--surface) !important;
        }

        /* Force readable text colors for all form fields */
        input, textarea, select,
        .stTextInput input,
        .stNumberInput input,
        [data-baseweb="input"] input,
        [data-baseweb="textarea"] textarea,
        [data-baseweb="select"] input {
            color: var(--text) !important;
            -webkit-text-fill-color: var(--text) !important;
            opacity: 1 !important;
            font-weight: 500 !important;
        }

        input::placeholder,
        textarea::placeholder {
            color: #7a7a7a !important;
            opacity: 1 !important;
        }

        /* Streamlit number input +/- controls */
        .stNumberInput button,
        [data-baseweb="input"] button {
            color: var(--text) !important;
            opacity: 1 !important;
        }

        .stButton > button,
        .stDownloadButton > button{
            border-radius: 12px !important;
            padding: 10px 18px !important;
            font-weight: 600 !important;
            border: none !important;
            background: var(--text) !important;
            color: var(--surface) !important;
            box-shadow: none !important;
        }

        .stButton > button *,
        .stDownloadButton > button * {
            color: var(--surface) !important;
            -webkit-text-fill-color: var(--surface) !important;
            opacity: 1 !important;
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover{
            filter: brightness(1.05);
        }

        section[data-testid="stFileUploaderDropzone"]{
            box-shadow: 0 0 0 1px var(--line) !important;
            border-radius: 16px !important;
            background: var(--surface) !important;
        }

        details{
            box-shadow: 0 0 0 1px var(--line) !important;
            border-radius: 14px !important;
            background: var(--surface) !important;
        }

        hr{
            border: none;
            border-top: 1px solid var(--line-soft);
            margin: 18px 0;
        }

        div[data-testid="stMetric"]{
            background: var(--surface);
            box-shadow: 0 0 0 1px var(--line);
            border-radius: 14px;
            padding: 16px;
            min-height: auto;              /* Biarkan grow sesuai konten */
            overflow: visible;             /* Tampilkan semua */
            height: auto;                  /* Tidak fixed height */
        }

        div[data-testid="stMetric"] [data-testid="stMetricValue"]{
            font-size: 20px;
            white-space: normal;           /* Wrap text */
            word-break: break-word;        /* Break long words */
            overflow-wrap: break-word;     /* Modern standard */
            overflow: visible;             /* Show all */
            max-width: 100%;               /* Full width */
            display: block;                /* Block element untuk wrap */
        }

        div[data-testid="stProgress"] > div{
            box-shadow: 0 0 0 1px var(--line);
            border-radius: 9999px;
            background: var(--surface);
            overflow: hidden;
        }

        div[data-testid="stProgress"] div[role="progressbar"]{
            background: var(--blue) !important;
        }

        div[data-testid="stDataFrame"]{
            box-shadow: 0 0 0 1px var(--line);
            border-radius: 14px;
            overflow: hidden;
            background: var(--surface);
        }

        div[data-testid="stAlert"]{
            border-radius: 14px;
            box-shadow: 0 0 0 1px var(--line);
        }

        button:focus, input:focus, select:focus, textarea:focus{
            outline: 2px solid var(--focus) !important;
            outline-offset: 2px;
        }

        /* Dark mode support */
        @media (prefers-color-scheme: dark){
            :root{
                --bg: #0f0f0f;
                --surface: #1a1a1a;
                --text: #ececec;
                --muted: #b0b0b0;
                --line: rgba(255,255,255,0.08);
                --line-soft: rgba(255,255,255,0.06);
                --shadow-soft: rgba(0,0,0,0.3);
                --badge-bg: #1e3a5f;
                --badge-text: #6da8ff;
            }
            
            .stApp{
                background:
                    radial-gradient(circle at top left, rgba(10,114,239,0.03), transparent 28%),
                    radial-gradient(circle at top right, rgba(222,29,141,0.02), transparent 22%),
                    linear-gradient(180deg, #0f0f0f 0%, #0f0f0f 100%);
                color: var(--text);
            }
            
            html, body, [class*="css"]{
                color: var(--text);
            }
        }

        @media (max-width: 900px){
            .hero-row,
            .content-grid,
            .workflow,
            .metric-strip,
            .hero-summary{
                grid-template-columns: 1fr;
            }
            .hero h1{
                font-size: clamp(34px, 10vw, 48px);
            }
        }

        @media (max-width: 600px){
            .block-container{ padding-left: 16px; padding-right: 16px; }
            .panel{ padding: 18px; }
            .hero{ padding-top: 0; }
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

apply_vercel_theme()


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
# Load Artefak Deployment (Multi-Model)
# =====================================================
@st.cache_resource
def load_shared_artifacts():
    """Load artefak yang digunakan bersama semua model."""
    feature_order_path = find_artifact("feature_order.json")
    imputer_path = find_artifact("feature_imputer.joblib")
    scaler_path = find_artifact("minmax_scaler.joblib")
    
    with open(feature_order_path, "r", encoding="utf-8") as f:
        feature_order = json.load(f)
    
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        old_imputer = joblib.load(imputer_path)
    
    # Rebuild imputer to avoid version-mismatch errors (e.g. _fill_dtype
    # renamed to _fit_dtype between scikit-learn 1.5 → 1.8).
    try:
        old_imputer.transform(
            pd.DataFrame(np.zeros((1, old_imputer.n_features_in_)),
                         columns=old_imputer.feature_names_in_)
        )
        imputer = old_imputer  # works fine, use as-is
    except AttributeError:
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(
            strategy=old_imputer.strategy,
            missing_values=old_imputer.missing_values,
            fill_value=old_imputer.fill_value,
        )
        # Fit on dummy data then overwrite statistics with the original values
        dummy = pd.DataFrame(
            np.zeros((2, old_imputer.n_features_in_)),
            columns=old_imputer.feature_names_in_,
        )
        imputer.fit(dummy)
        imputer.statistics_ = old_imputer.statistics_
    
    scaler = joblib.load(scaler_path)
    
    return imputer, scaler, feature_order


@st.cache_resource
def load_model_registry():
    """Load registry untuk semua model tersedia."""
    try:
        registry_path = find_artifact("model_registry.json")
        with open(registry_path, "r", encoding="utf-8") as f:
            registry = json.load(f)
        return registry
    except FileNotFoundError:
        st.error("model_registry.json tidak ditemukan.")
        st.stop()


@st.cache_resource
def load_xgb_model(model_json_path: str):
    """Load model XGBoost dari file JSON."""
    model = xgb.Booster()
    model.load_model(str(model_json_path))
    return model


def get_available_models():
    """Dapatkan daftar model tersedia dari registry."""
    registry = load_model_registry()
    models_dict = registry.get("models", {})
    return {
        key: info["display_name"] 
        for key, info in models_dict.items()
    }




@st.cache_data
def load_metrics_registry():
    metrics_path = find_artifact("model_metrics_registry.csv")
    return pd.read_csv(metrics_path)


try:
    imputer, scaler, feature_order = load_shared_artifacts()
    registry = load_model_registry()
    available_models = get_available_models()
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
def build_input_df(input_values: dict, feature_order: list[str]) -> pd.DataFrame:
    """Menyusun input user sesuai urutan fitur saat training."""
    row = {feature: input_values.get(feature, np.nan) for feature in feature_order}
    return pd.DataFrame([row], columns=feature_order)


def predict_xgb_booster(model, dmatrix, best_iteration=None):
    """
    Prediksi XGBoost Booster dengan best_iteration jika tersedia.
    best_iteration digunakan untuk memastikan prediksi konsisten dengan evaluasi.
    """
    if best_iteration is not None and best_iteration >= 0:
        try:
            return model.predict(dmatrix, iteration_range=(0, int(best_iteration) + 1))
        except (TypeError, Exception):
            pass
    return model.predict(dmatrix)


def predict_quality(input_df: pd.DataFrame, model, best_iteration=None):
    """Melakukan preprocessing dan prediksi menggunakan artefak hasil training."""
    imputed = imputer.transform(input_df)
    scaled = scaler.transform(imputed)
    dmatrix = xgb.DMatrix(scaled, feature_names=feature_order)
    
    probability = float(predict_xgb_booster(model, dmatrix, best_iteration)[0])
    threshold = float(registry.get("prediction_threshold", 0.5))
    predicted_class = int(probability > threshold)
    
    class_mapping = registry.get("class_mapping", {"0": "Tidak Layak Minum", "1": "Layak Minum"})
    label = class_mapping.get(str(predicted_class), str(predicted_class))
    return predicted_class, label, probability, threshold


# =====================================================
# Sidebar: Navigasi & Model Selection
# =====================================================
with st.sidebar:
    st.markdown("### 📊 Pilih Model")
    st.markdown(
        '<p class="section-lead">Terdapat 6 model tersedia dengan kombinasi split data dan metode tuning.</p>',
        unsafe_allow_html=True,
    )
    
    # Group models by split and model_type
    model_options_by_split = {}
    for model_key, display_name in available_models.items():
        model_info = registry["models"][model_key]
        split = model_info.get("split", "unknown")
        if split not in model_options_by_split:
            model_options_by_split[split] = []
        model_options_by_split[split].append((model_key, display_name, model_info))
    
    # Sort splits
    sorted_splits = sorted(model_options_by_split.keys())
    
    selected_split = st.selectbox(
        "Pilih Split Data:",
        sorted_splits,
        key="split_selector"
    )
    
    models_in_split = model_options_by_split[selected_split]
    selected_model_key = st.selectbox(
        "Pilih Model:",
        [key for key, _, _ in models_in_split],
        format_func=lambda key: next((name for k, name, _ in models_in_split if k == key), key),
        key="model_selector"
    )
    
    selected_model_info = registry["models"][selected_model_key]
    
    # Display model details
    st.divider()
    st.markdown("### 📈 Detail Model")
    st.write(f"**Key:** `{selected_model_key}`")
    st.write(f"**Tipe:** {selected_model_info.get('model_type', 'N/A').upper()}")
    st.write(f"**Split:** {selected_model_info.get('split', 'N/A')}")
    
    st.markdown("#### Metrik")
    metrics = selected_model_info.get("metrics", {})
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Accuracy", f"{metrics.get('accuracy', 0):.4f}")
        st.metric("Precision", f"{metrics.get('precision', 0):.4f}")
    with col2:
        st.metric("Recall", f"{metrics.get('recall', 0):.4f}")
        st.metric("F1-Score", f"{metrics.get('f1_score', 0):.4f}")
    
    st.markdown("#### Training Info")
    st.write(f"**Best Iteration:** {selected_model_info.get('best_iteration', 'N/A')}")
    st.write(f"**Best Score:** {selected_model_info.get('best_score', 'N/A'):.6f}")
    st.write(f"**Training Time:** {selected_model_info.get('training_time_seconds', 'N/A'):.2f}s")

# Load selected model
model_json_path = find_artifact(selected_model_info["model_json"])
selected_model = load_xgb_model(model_json_path)
best_iteration = selected_model_info.get("best_iteration")




# =====================================================
# Halaman Prediksi
# =====================================================
def render_prediction():
    st.markdown(
        """
        <section class="hero">
            <span class="eyebrow">Prediksi Kualitas Air</span>
            <h1>Masukkan Parameter, Dapatkan Hasil</h1>
            <p>
            Halaman ini difokuskan untuk prediksi tunggal secepat mungkin.
            Isi nilai parameter di kiri, lalu lihat hasil prediksi di kanan.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    input_col, result_col = st.columns([1.2, 0.8], gap="large")

    submitted = False
    input_df = None
    predicted_class = None
    label = ""
    probability = 0.0
    threshold = float(registry.get("prediction_threshold", 0.5))

    with input_col:
        st.subheader("1) Input Parameter")
        st.markdown(
            '<p class="section-lead">Isi nilai parameter berikut sesuai kondisi sampel air yang ingin diprediksi.</p>',
            unsafe_allow_html=True,
        )

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

            submitted = st.form_submit_button("Prediksi Sekarang")

        st.markdown("</div>", unsafe_allow_html=True)

    with result_col:
        st.subheader("2) Hasil Prediksi")

        if submitted:
            input_df = build_input_df(input_values, feature_order)
            predicted_class, label, probability, threshold = predict_quality(
                input_df,
                selected_model,
                best_iteration
            )

            if predicted_class == 1:
                st.success(f"Prediksi: {label}")
            else:
                st.error(f"Prediksi: {label}")

            st.metric("Probabilitas layak minum", f"{probability:.4f}")
            st.progress(min(max(probability, 0.0), 1.0))

            d1, d2 = st.columns(2, gap="small")
            with d1:
                st.markdown(
                    f"""
                    <div class="summary-card">
                        <strong>Threshold</strong>
                        <span>{threshold}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with d2:
                st.markdown(
                    f"""
                    <div class="summary-card">
                        <strong>Label</strong>
                        <span>{label}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("Belum ada hasil. Isi parameter lalu klik **Prediksi Sekarang**.")

        st.markdown("</div>", unsafe_allow_html=True)

    if submitted and input_df is not None:
        with st.expander("Lihat detail input dan preprocessing"):
            st.write("**Input sesuai feature order:**")
            st.dataframe(input_df, use_container_width=True)

        # ── Dashboard ditampilkan di bawah hasil prediksi ──
        st.divider()
        metrics_df = load_metrics_registry()

        st.markdown(
            """
            <section class="hero">
                <span class="eyebrow">Dashboard</span>
                <h1>Performa Model</h1>
                <p>
                Ringkasan metrik dari semua model yang tersedia di registry.
                </p>
            </section>
            """,
            unsafe_allow_html=True,
        )

        cols_to_hide = ["model_key", "model_json", "model_joblib"]
        metrics_display = metrics_df.drop(columns=cols_to_hide, errors="ignore")

        # Ringkasan best model
        best_row = metrics_df.sort_values("f1_score", ascending=False).iloc[0]

        c1, c2, c3, c4, c5 = st.columns([2, 1, 1, 1, 1])

        c1.metric(
            "Best Model",
            best_row["display_name"]
        )

        c2.metric(
            "Accuracy",
            f"{best_row['accuracy']:.4f}"
        )

        c3.metric(
            "Precision",
            f"{best_row['precision']:.4f}"
        )

        c4.metric(
            "Recall",
            f"{best_row['recall']:.4f}"
        )

        c5.metric(
            "F1-Score",
            f"{best_row['f1_score']:.4f}"
        )

        sst.markdown("### Perbandingan Performa 6 Model")
        # =========================
        # Data Metrics
        # =========================
        comparison_chart_df = metrics_df[
            ["display_name", "accuracy", "precision", "recall", "f1_score"]
        ].copy()

        # Rename agar lebih rapi di chart
        comparison_chart_df = comparison_chart_df.rename(
            columns={
                "display_name": "Model",
                "accuracy": "Accuracy",
                "precision": "Precision",
                "recall": "Recall",
                "f1_score": "F1-Score",
            }
        )

        # Ubah format supaya horizontal grouped bar chart
        chart_data = comparison_chart_df.melt(
            id_vars="Model",
            var_name="Metric",
            value_name="Score"
        )

        # =========================
        # Chart
        # =========================
        st.bar_chart(
            data=chart_data,
            x="Metric",
            y="Score",
            color="Model",
            horizontal=True,
            use_container_width=True
        )

        st.markdown("### Perbandingan Accuracy")
        acc_df = metrics_df[["display_name", "accuracy"]].set_index("display_name")
        st.bar_chart(acc_df)

        st.markdown("### Confusion Matrix (Semua Model)")
        for model_key, model_info in registry.get("models", {}).items():
            cm = model_info.get("confusion_matrix", {})
            st.markdown(f"**{model_info.get('display_name', model_key)}**")
            if cm:
                cm_df = pd.DataFrame(
                    [
                        [cm.get("tn", 0), cm.get("fp", 0)],
                        [cm.get("fn", 0), cm.get("tp", 0)],
                    ],
                    index=["Actual: Tidak Layak", "Actual: Layak"],
                    columns=["Pred: Tidak Layak", "Pred: Layak"],
                )
                st.dataframe(cm_df, use_container_width=True)
            else:
                st.info("Confusion matrix belum tersedia untuk model ini.")


# =====================================================
# Render Halaman Prediksi
# =====================================================
render_prediction()

