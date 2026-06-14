
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

# ======================
# PAGE CONFIG
# ======================

st.set_page_config(
    page_title="Cardiovascular Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

# ======================
# LOAD DATA
# ======================

df = pd.read_csv("cardio_train.csv", sep=";")

df["age_years"] = (df["age"] / 365).astype(int)
df["BMI"] = df["weight"] / ((df["height"]/100)**2)

df["age_group"] = pd.cut(
    df["age_years"],
    bins=[30,40,50,60,70],
    labels=["30-40","40-50","50-60","60-70"]
)

# ======================
# LOAD MODEL
# ======================

try:
    model = joblib.load("cardio_model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
except:
    model = None

# ======================
# SIDEBAR
# ======================

st.sidebar.title("❤️ Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "Overview",
        "EDA Dashboard",
        "Model Performance",
        "Prediction"
    ]
)

# ======================
# FILTERS
# ======================

st.sidebar.markdown("---")
st.sidebar.subheader("Filters")

gender_filter = st.sidebar.multiselect(
    "Gender",
    sorted(df["gender"].unique()),
    default=sorted(df["gender"].unique())
)

chol_filter = st.sidebar.multiselect(
    "Cholesterol",
    sorted(df["cholesterol"].unique()),
    default=sorted(df["cholesterol"].unique())
)

cardio_filter = st.sidebar.multiselect(
    "Disease Status",
    sorted(df["cardio"].unique()),
    default=sorted(df["cardio"].unique())
)

filtered_df = df[
    (df["gender"].isin(gender_filter))
    &
    (df["cholesterol"].isin(chol_filter))
    &
    (df["cardio"].isin(cardio_filter))
]

# ======================
# OVERVIEW PAGE
# ======================

if page == "Overview":

    st.title("❤️ Cardiovascular Disease Prediction Dashboard")

    c1,c2,c3,c4 = st.columns(4)

    c1.metric(
        "Patients",
        len(filtered_df)
    )

    c2.metric(
        "Average Age",
        round(filtered_df["age_years"].mean(),1)
    )

    c3.metric(
        "Average BMI",
        round(filtered_df["BMI"].mean(),1)
    )

    c4.metric(
        "Disease Rate",
        f"{filtered_df['cardio'].mean()*100:.1f}%"
    )

    st.markdown("---")

    st.subheader("Dataset Preview")

    st.dataframe(filtered_df.head())

    st.subheader("Statistical Summary")

    st.dataframe(filtered_df.describe())

# ======================
# EDA PAGE
# ======================

elif page == "EDA Dashboard":

    st.title("📊 Exploratory Data Analysis")

    fig1 = px.histogram(
        filtered_df,
        x="age_years",
        nbins=30,
        title="Age Distribution"
    )

    st.plotly_chart(
        fig1,
        use_container_width=True
    )

    fig2 = px.histogram(
        filtered_df,
        x="BMI",
        color="cardio",
        title="BMI Distribution"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    age_cardio = pd.crosstab(
        filtered_df["age_group"],
        filtered_df["cardio"],
        normalize="index"
    ) * 100

    st.subheader(
        "Disease Rate by Age Group"
    )

    st.bar_chart(age_cardio)

    st.subheader(
        "Correlation Heatmap"
    )

    corr = filtered_df.corr(
        numeric_only=True
    )

    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ======================
# MODEL PAGE
# ======================

elif page == "Model Performance":

    st.title("🤖 Model Performance")

    try:

        importance = pd.DataFrame({
            "Feature": feature_columns,
            "Importance": model.feature_importances_
        })

        importance = importance.sort_values(
            "Importance",
            ascending=False
        )

        st.subheader(
            "Top Important Features"
        )

        fig = px.bar(
            importance.head(10),
            x="Importance",
            y="Feature",
            orientation="h"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    except:
        st.warning(
            "Feature Importance Not Available"
        )

# ======================
# PREDICTION PAGE
# ======================

elif page == "Prediction":

    st.title(
        "🩺 Predict Cardiovascular Disease"
    )

    age = st.number_input(
        "Age",
        30,
        80,
        50
    )

    gender = st.selectbox(
        "Gender",
        [1,2]
    )

    height = st.number_input(
        "Height (cm)",
        100,
        250,
        170
    )

    weight = st.number_input(
        "Weight (kg)",
        30,
        200,
        75
    )

    ap_hi = st.number_input(
        "Systolic BP",
        80,
        250,
        120
    )

    ap_lo = st.number_input(
        "Diastolic BP",
        40,
        200,
        80
    )

    cholesterol = st.selectbox(
        "Cholesterol",
        [1,2,3]
    )

    gluc = st.selectbox(
        "Glucose",
        [1,2,3]
    )

    smoke = st.selectbox(
        "Smoke",
        [0,1]
    )

    alco = st.selectbox(
        "Alcohol",
        [0,1]
    )

    active = st.selectbox(
        "Active",
        [0,1]
    )

    bmi = weight / ((height/100)**2)

    st.info(
        f"BMI = {bmi:.2f}"
    )

    if st.button(
        "Predict Disease"
    ):

        if model is None:

            st.error(
                "Model files not found."
            )

        else:

            patient = pd.DataFrame({
                "id":[0],
                "age":[age*365],
                "gender":[gender],
                "height":[height],
                "weight":[weight],
                "ap_hi":[ap_hi],
                "ap_lo":[ap_lo],
                "cholesterol":[cholesterol],
                "gluc":[gluc],
                "smoke":[smoke],
                "alco":[alco],
                "active":[active],
                "age_years":[age],
                "BMI":[bmi]
            })

            patient = patient[feature_columns]

            patient_scaled = scaler.transform(
                patient
            )

            prediction = model.predict(
                patient_scaled
            )[0]

            probability = model.predict_proba(
                patient_scaled
            )[0][1]

            if prediction == 1:

                st.error(
                    f"High Risk ({probability:.1%})"
                )

            else:

                st.success(
                    f"Low Risk ({1-probability:.1%})"
                )
Update app.py
