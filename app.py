import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    layout="wide"
)

# -----------------------------
# STYLING (FROM YOUR REFERENCE)
# -----------------------------
st.markdown("""
<style>

/* ===== GLOBAL BACKGROUND ===== */
.stApp {
    background: linear-gradient(
        135deg,
        #f0f4ff,
        #e6f7f1,
        #fff7e6
    );
    background-attachment: fixed;
    padding: 100px;
}

/* ===== TITLE SPACING ===== */
h1 {
    margin-top: 2.5rem !important;
}

/* ===== SIDEBAR FIX ===== */
section[data-testid="stSidebar"] {
    height: 100vh;
    overflow-y: auto !important;
    padding-bottom: 2rem;
}

/* ===== MAIN CONTENT CARD ===== */
.block-container {
    background: rgba(255, 255, 255, 0.75);
    backdrop-filter: blur(10px);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    box-shadow: 0 10px 30px rgba(0,0,0,0.08);
}

/* ===== SIDEBAR GRADIENT ===== */
section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #1f3c88,
        #2a5298,
        #1e3c72
    );
    color: white;
}

/* ===== SIDEBAR LABELS ===== */
section[data-testid="stSidebar"] label {
    font-size: 16px !important;
    font-weight: 700 !important;
    color: #ffffff !important;
}

/* ===== SLIDER VALUES ===== */
section[data-testid="stSidebar"] .stSlider span {
    font-size: 15px !important;
    font-weight: 600 !important;
    color: #ffdddd !important;
}

/* ===== BUTTON ===== */
.stButton>button {
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    color: white;
    border-radius: 12px;
    padding: 0.6rem 1.4rem;
    font-size: 16px;
    font-weight: 600;
    border: none;
    transition: 0.3s ease;
}

.stButton>button:hover {
    transform: scale(1.03);
    box-shadow: 0 6px 18px rgba(0,0,0,0.2);
}

/* ===== ALERTS ===== */
div.stAlert-success {
    background: linear-gradient(90deg, #e0f8e9, #c6f6d5);
    border-radius: 10px;
}

div.stAlert-error {
    background: linear-gradient(90deg, #ffe0e0, #ffbdbd);
    border-radius: 10px;
}

/* ===== FOOTER ===== */
footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# APP TITLE & DESCRIPTION
# -----------------------------
st.title("🟢 Customer Segmentation Dashboard")
st.write(
    "This system uses **K-Means Clustering** to group customers "
    "based on their purchasing behavior and similarities."
)
st.markdown("👉 *Discover hidden customer groups without predefined labels.*")

# -----------------------------
# FILE UPLOAD
# -----------------------------
uploaded_file = st.file_uploader(
    "📂 Upload Customer Dataset (CSV)", type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    # -----------------------------
    # NUMERICAL FEATURES
    # -----------------------------
    numeric_features = df.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    # -----------------------------
    # SIDEBAR CONTROLS
    # -----------------------------
    st.sidebar.header("🔧 Clustering Controls")

    feature_1 = st.sidebar.selectbox(
        "Select Feature 1", numeric_features
    )

    feature_2 = st.sidebar.selectbox(
        "Select Feature 2",
        [f for f in numeric_features if f != feature_1]
    )

    k = st.sidebar.slider(
        "Number of Clusters (K)", 2, 10, 3
    )

    random_state = st.sidebar.number_input(
        "Random State (Optional)", value=42, step=1
    )

    run_btn = st.sidebar.button("🟦 Run Clustering")

    # -----------------------------
    # RUN CLUSTERING
    # -----------------------------
    if run_btn:

        X = df[[feature_1, feature_2]]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        kmeans = KMeans(
            n_clusters=k,
            random_state=random_state
        )
        df["Cluster"] = kmeans.fit_predict(X_scaled)

        # -----------------------------
        # VISUALIZATION
        # -----------------------------
        st.subheader("📊 Customer Clusters Visualization")

        fig, ax = plt.subplots()
        ax.scatter(
            df[feature_1],
            df[feature_2],
            c=df["Cluster"]
        )

        centers = scaler.inverse_transform(
            kmeans.cluster_centers_
        )
        ax.scatter(
            centers[:, 0],
            centers[:, 1],
            s=300,
            c="black",
            marker="X",
            label="Cluster Centers"
        )

        ax.set_xlabel(feature_1)
        ax.set_ylabel(feature_2)
        ax.set_title("K-Means Clustering Result")
        ax.legend()

        st.pyplot(fig)

        # -----------------------------
        # CLUSTER SUMMARY
        # -----------------------------
        st.subheader("📋 Cluster Summary")

        summary = (
            df.groupby("Cluster")
            .agg(
                Count=("Cluster", "count"),
                Avg_Feature_1=(feature_1, "mean"),
                Avg_Feature_2=(feature_2, "mean")
            )
            .reset_index()
        )

        st.dataframe(summary)

        # -----------------------------
        # BUSINESS INTERPRETATION
        # -----------------------------
        st.subheader("💡 Business Interpretation")

        for _, row in summary.iterrows():
            cid = int(row["Cluster"])

            if (
                row["Avg_Feature_1"] > summary["Avg_Feature_1"].mean()
                and row["Avg_Feature_2"] > summary["Avg_Feature_2"].mean()
            ):
                msg = "High-spending customers across multiple categories"
            elif (
                row["Avg_Feature_1"] < summary["Avg_Feature_1"].mean()
                and row["Avg_Feature_2"] < summary["Avg_Feature_2"].mean()
            ):
                msg = "Budget-conscious customers with low annual spend"
            else:
                msg = "Moderate spenders with selective purchasing behavior"

            st.write(f"🟢 **Cluster {cid}:** {msg}")

        # -----------------------------
        # USER GUIDANCE
        # -----------------------------
        st.info(
            "Customers in the same cluster exhibit similar purchasing behaviour "
            "and can be targeted with similar business strategies."
        )

else:
    st.warning("Please upload a CSV dataset to begin clustering.")
