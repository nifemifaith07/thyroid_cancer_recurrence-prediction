
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ---------- 1. Load artefacts ----------
artefacts = joblib.load("thyroid_recurrence_detection_lr.pkl")
model    = artefacts["model"]
encoder  = artefacts["encoder"]
scaler   = artefacts["scaler"]
exp_cols = artefacts["columns"]      # training-time order

# ---------- 2. Streamlit UI ----------
st.set_page_config(page_title="Thyroid Recurrence Predictor")
st.title("🧠 Thyroid Cancer Recurrence Risk")

with st.form("patient_form"):
    st.subheader("Patient information")
    age  = st.number_input("Age", 1, 120, value=45)

    gender             = st.selectbox("Gender", ["M", "F"])
    smoking            = st.selectbox("Smoking", ["No", "Yes"])
    hx_smoking         = st.selectbox("Hx Smoking", ["No", "Yes"])
    hx_radiothreapy    = st.selectbox("Hx Radiotherapy", ["No", "Yes"])
    thyroid_function   = st.selectbox("Thyroid Function", [
        "Euthyroid", "Clinical Hyperthyroidism", "Clinical Hypothyroidism",
        "Subclinical Hyperthyroidism", "Subclinical Hypothyroidism"])
    physical_examination      = st.selectbox("Physical Examination", [
        "Normal", "Single nodular goiter-left", "Single nodular goiter-right",
        "Multinodular goiter", "Diffuse goiter"])
    adenopathy         = st.selectbox("Adenopathy", [
        "No", "Left", "Right", "Bilateral", "Posterior", "Extensive"])
    pathology          = st.selectbox("Pathology", [
        "Micropapillary", "Papillary", "Follicular", "Hurthel cell"])
    focality           = st.selectbox("Focality", ["Unifocal", "Multifocal"])
    risk               = st.selectbox("Risk", ["Low", "Intermediate", "High"])
    tumor              = st.selectbox("Tumor (T)", ["T1a", "T1b", "T2", "T3a", "T3b", "T4a", "T4b"])
    lymph_nodes        = st.selectbox("Nodes (N)", ["N0", "N1a", "N1b"])
    cancer_metastasis  = st.selectbox("Metastasis (M)", ["M0", "M1"])
    stage              = st.selectbox("Stage", ["I", "II", "III", "IVA", "IVB"])
    treatment_response = st.selectbox("Treatment Response", [
        "Indeterminate", "Excellent", "Structural Incomplete", "Biochemical Incomplete"])

    submitted = st.form_submit_button("Predict")

# ---------- 3. Pre-process & predict ----------
if submitted:
    # 3-a. Build a 1-row DataFrame in the *original* raw format
    raw = pd.DataFrame([{
        "age": age,
        "gender": gender,
        "smoking": smoking,
        "hx_smoking": hx_smoking,
        "hx_radiothreapy": hx_radiothreapy,
        "thyroid_function": thyroid_function,
        "physical_examination": physical_examination,
        "adenopathy": adenopathy,
        "pathology": pathology,
        "focality": focality,
        "risk": risk,
        "tumor": tumor,
        "lymph_nodes": lymph_nodes,
        "cancer_metastasis": cancer_metastasis,
        "stage": stage,
        "treatment_response": treatment_response
    }])

    # 3-b. One-hot encode categoricals
    cat_cols = encoder.feature_names_in_.tolist()
    encoded  = pd.DataFrame(
        encoder.transform(raw[cat_cols]),
        columns = encoder.get_feature_names_out(cat_cols)
    )

    # 3-c. Scale age
    age_scaled = pd.DataFrame(
        scaler.transform(raw[['age']]),
        columns = ['age']
    )

    # 3-d. Re-assemble and align to training column order
    processed = pd.concat([age_scaled, encoded], axis=1)
    # fill any columns that may be missing in new data
    missing = set(exp_cols) - set(processed.columns)
    for col in missing:
        processed[col] = 0
    processed = processed[exp_cols]

    # 3-e. Predict
    prob = model.predict_proba(processed)[0,1]
    pred = model.predict(processed)[0]

    st.markdown("---")
    st.subheader("📊 Prediction")
    st.write(f"**Probability of recurrence:** `{prob:.1%}`")
    st.write(f"**Predicted class:** {'Yes' if pred == 1 else 'No'}")
