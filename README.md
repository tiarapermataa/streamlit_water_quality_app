# Streamlit Prediksi Kualitas Air

Draft aplikasi Streamlit untuk menjalankan model XGBoost + Optuna hasil training notebook.

## Struktur Folder

Letakkan file seperti ini:

```text
streamlit_water_quality_app/
├── app.py
├── requirements.txt
└── deployment_artifacts/
    ├── xgboost_model.json
    ├── xgboost_model.joblib              # opsional
    ├── feature_imputer.joblib
    ├── minmax_scaler.joblib
    ├── feature_order.json
    └── deployment_metadata.json
```

File di folder `deployment_artifacts` dibuat dari cell export di notebook training.

## Cara Menjalankan

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Catatan

Aplikasi ini memakai urutan inferensi berikut:

1. Load `feature_order.json`.
2. Susun input user sesuai urutan fitur saat training.
3. Transform input dengan `feature_imputer.joblib`.
4. Transform input dengan `minmax_scaler.joblib`.
5. Prediksi menggunakan `xgboost_model.json`.

Hasil prediksi tidak menggantikan pengujian laboratorium resmi.
