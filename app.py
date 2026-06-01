import os
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

MODEL_PATH = "fraud_model.pkl"
SCALER_PATH = "scaler.pkl"
METRICS_PATH = "model_metrics.pkl"

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
)


def check_required_files():
    """Check whether trained model files exist."""
    missing_files = []

    for file_path in [MODEL_PATH, SCALER_PATH, METRICS_PATH]:
        if not os.path.exists(file_path):
            missing_files.append(file_path)

    return missing_files


@st.cache_resource
def load_artifacts():
    """Load model, scaler, and metrics."""
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    metrics = joblib.load(METRICS_PATH)
    return model, scaler, metrics


def scale_transaction(dataframe, scaler):
    """Scale Time and Amount columns."""
    dataframe = dataframe.copy()
    dataframe[["Time", "Amount"]] = scaler.transform(dataframe[["Time", "Amount"]])
    return dataframe


def get_risk_category(probability):
    """Return risk category according to probability."""
    if probability >= 0.70:
        return "High Risk", "🚨"
    if probability >= 0.40:
        return "Medium Risk", "⚠️"
    return "Low Risk", "✅"


def single_prediction_page(model, scaler, feature_columns):
    st.header("Single Transaction Prediction")

    st.info(
        "Enter transaction details manually. V1 to V28 are anonymized PCA features "
        "from the dataset."
    )

    col1, col2 = st.columns(2)

    with col1:
        time = st.number_input("Time", min_value=0.0, value=10000.0, step=100.0)

    with col2:
        amount = st.number_input("Amount", min_value=0.0, value=100.0, step=10.0)

    st.subheader("Transaction Features")

    values = {}
    columns = st.columns(4)

    for i in range(1, 29):
        with columns[(i - 1) % 4]:
            values[f"V{i}"] = st.number_input(
                f"V{i}",
                value=0.0,
                format="%.6f",
            )

    input_data = pd.DataFrame(
        [
            {
                "Time": time,
                **values,
                "Amount": amount,
            }
        ]
    )

    input_data = input_data[feature_columns]

    st.subheader("Input Preview")
    st.dataframe(input_data, use_container_width=True)

    if st.button("Predict Transaction", type="primary"):
        scaled_input = scale_transaction(input_data, scaler)

        prediction = model.predict(scaled_input)[0]
        probability = model.predict_proba(scaled_input)[0][1]

        risk, icon = get_risk_category(probability)

        st.subheader("Prediction Result")

        if prediction == 1:
            st.error(f"{icon} Fraudulent Transaction Detected")
        else:
            st.success(f"{icon} Genuine Transaction")

        st.metric("Fraud Probability", f"{probability * 100:.2f}%")
        st.write(f"Risk Category: **{risk}**")


def bulk_prediction_page(model, scaler, feature_columns):
    st.header("Bulk CSV Prediction")

    st.write(
        "Upload a CSV file with columns Time, V1 to V28, and Amount. "
        "The app will predict fraud for all transactions."
    )

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file is None:
        st.warning("Upload a CSV file to continue.")
        return

    try:
        data = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Data Preview")
        st.dataframe(data.head(), use_container_width=True)

        missing_columns = [column for column in feature_columns if column not in data.columns]

        if missing_columns:
            st.error(f"Missing required columns: {missing_columns}")
            return

        input_data = data[feature_columns].copy()
        scaled_data = scale_transaction(input_data, scaler)

        predictions = model.predict(scaled_data)
        probabilities = model.predict_proba(scaled_data)[:, 1]

        result = data.copy()
        result["Fraud_Prediction"] = predictions
        result["Fraud_Probability"] = probabilities
        result["Result"] = result["Fraud_Prediction"].map(
            {
                0: "Genuine",
                1: "Fraud",
            }
        )

        total = len(result)
        fraud = int((result["Fraud_Prediction"] == 1).sum())
        genuine = total - fraud

        st.subheader("Prediction Summary")

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Transactions", total)
        col2.metric("Genuine Transactions", genuine)
        col3.metric("Fraud Transactions", fraud)

        st.subheader("Prediction Result")
        st.dataframe(result, use_container_width=True)

        csv_data = result.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Prediction Report",
            data=csv_data,
            file_name="fraud_prediction_report.csv",
            mime="text/csv",
        )

    except Exception as error:
        st.error(f"Error while processing file: {error}")


def dashboard_page(metrics):
    st.header("Model Performance Dashboard")

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Accuracy", f"{metrics['accuracy'] * 100:.2f}%")
    col2.metric("Precision", f"{metrics['precision'] * 100:.2f}%")
    col3.metric("Recall", f"{metrics['recall'] * 100:.2f}%")
    col4.metric("F1 Score", f"{metrics['f1_score'] * 100:.2f}%")
    col5.metric("ROC-AUC", f"{metrics['roc_auc'] * 100:.2f}%")

    st.subheader("Confusion Matrix")

    cm = metrics["confusion_matrix"]

    figure, axis = plt.subplots()
    axis.imshow(cm)
    axis.set_title("Confusion Matrix")
    axis.set_xlabel("Predicted Label")
    axis.set_ylabel("Actual Label")
    axis.set_xticks([0, 1])
    axis.set_yticks([0, 1])
    axis.set_xticklabels(["Genuine", "Fraud"])
    axis.set_yticklabels(["Genuine", "Fraud"])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axis.text(j, i, cm[i, j], ha="center", va="center")

    st.pyplot(figure)

    st.subheader("Classification Report")
    st.code(metrics["classification_report"])


def about_page():
    st.header("About Project")

    st.write(
        """
        This is a Machine Learning based Credit Card Fraud Detection application.
        It predicts whether a transaction is genuine or fraudulent using transaction features.
        """
    )

    st.subheader("Technology Used")

    st.write(
        """
        - Python
        - Pandas
        - NumPy
        - Scikit-learn
        - Random Forest Classifier
        - Streamlit
        - Joblib
        - Matplotlib
        """
    )

    st.subheader("Important Note")

    st.write(
        """
        The dataset uses anonymized PCA features V1 to V28. Because original banking features
        are hidden for privacy, this project is best for learning, college project submission,
        and demo purposes.
        """
    )


def main():
    st.title("💳 Credit Card Fraud Detection App")
    st.caption("Machine Learning project using Random Forest and Streamlit")

    missing_files = check_required_files()

    if missing_files:
        st.error("Model files are missing.")
        st.write("First place creditcard.csv in this folder, then run:")
        st.code("python train_model.py")
        st.write("Missing files:")
        st.write(missing_files)
        return

    model, scaler, metrics = load_artifacts()
    feature_columns = metrics["feature_columns"]

    st.sidebar.title("Navigation")

    page = st.sidebar.radio(
        "Choose Page",
        [
            "Single Prediction",
            "Bulk CSV Prediction",
            "Model Dashboard",
            "About Project",
        ],
    )

    if page == "Single Prediction":
        single_prediction_page(model, scaler, feature_columns)
    elif page == "Bulk CSV Prediction":
        bulk_prediction_page(model, scaler, feature_columns)
    elif page == "Model Dashboard":
        dashboard_page(metrics)
    elif page == "About Project":
        about_page()


if __name__ == "__main__":
    main()
