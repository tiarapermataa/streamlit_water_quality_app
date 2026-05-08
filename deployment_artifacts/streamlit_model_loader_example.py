import os
import json
import joblib
import pandas as pd
import xgboost as xgb
import streamlit as st

ARTIFACT_DIR = "deployment_artifacts"

@st.cache_resource
def load_shared_artifacts():
    with open(os.path.join(ARTIFACT_DIR, "model_registry.json"), "r", encoding="utf-8") as f:
        registry = json.load(f)

    with open(os.path.join(ARTIFACT_DIR, registry["feature_order_file"]), "r", encoding="utf-8") as f:
        feature_order = json.load(f)

    input_schema = pd.read_csv(os.path.join(ARTIFACT_DIR, registry["input_schema_file"]))

    imputer = joblib.load(os.path.join(ARTIFACT_DIR, registry["imputer_file"]))
    scaler = joblib.load(os.path.join(ARTIFACT_DIR, registry["scaler_file"]))

    return registry, feature_order, input_schema, imputer, scaler

@st.cache_resource
def load_selected_model(model_path):
    model = xgb.Booster()
    model.load_model(model_path)
    return model

def predict_with_best_iteration(model, dmatrix, best_iteration=None):
    if best_iteration is not None:
        try:
            return model.predict(dmatrix, iteration_range=(0, int(best_iteration) + 1))
        except TypeError:
            pass
    return model.predict(dmatrix)

registry, feature_order, input_schema, imputer, scaler = load_shared_artifacts()

model_options = list(registry["models"].keys())
selected_model_key = st.selectbox(
    "Pilih model",
    model_options,
    format_func=lambda key: registry["models"][key]["display_name"]
)

selected_model_info = registry["models"][selected_model_key]
selected_model_path = os.path.join(ARTIFACT_DIR, selected_model_info["model_json"])
model = load_selected_model(selected_model_path)

st.write("Model digunakan:", selected_model_info["display_name"])
st.write("Metrics:", selected_model_info["metrics"])

schema_map = input_schema.set_index("feature").to_dict(orient="index")

input_values = {}
for feature in feature_order:
    stats = schema_map.get(feature, {})
    default_value = float(stats.get("median", 0.0))
    min_value = float(stats.get("min", 0.0))
    max_value = float(stats.get("max", 0.0))

    input_values[feature] = st.number_input(
        feature,
        min_value=min_value,
        max_value=max_value,
        value=default_value
    )

if st.button("Prediksi"):
    input_df = pd.DataFrame([input_values], columns=feature_order)

    input_imputed_array = imputer.transform(input_df)
    input_imputed_df = pd.DataFrame(input_imputed_array, columns=feature_order)

    input_scaled = scaler.transform(input_imputed_df)

    dmatrix = xgb.DMatrix(input_scaled, feature_names=feature_order)

    probability = float(
        predict_with_best_iteration(
            model,
            dmatrix,
            best_iteration=selected_model_info.get("best_iteration")
        )[0]
    )

    threshold = float(selected_model_info.get("prediction_threshold", 0.5))
    prediction = int(probability > threshold)

    label = registry["class_mapping"][str(prediction)]

    st.metric("Probabilitas Layak Minum", f"{probability:.4f}")
    st.success(f"Hasil prediksi: {label}")
