import streamlit as st
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import numpy as np

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="DBSCAN Clustering", layout="centered")

st.title("DBSCAN Clustering Application")
st.write("Simple Input → Output Interface")

# -----------------------------
# Load Dataset (Hidden)
# -----------------------------
wine = load_wine()
X = wine.data

# Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# User Inputs
# -----------------------------
st.subheader("Input Parameters")

eps = st.number_input(
    "Enter eps (Neighborhood radius)",
    min_value=0.1,
    max_value=10.0,
    value=2.0,
    step=0.1
)

min_samples = st.number_input(
    "Enter min_samples (Minimum points)",
    min_value=1,
    max_value=20,
    value=2
)

# -----------------------------
# Run DBSCAN
# -----------------------------
if st.button("Run DBSCAN"):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X_scaled)

    unique_labels = set(labels)
    total_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    noise_points = np.sum(labels == -1)

    # -----------------------------
    # Output
    # -----------------------------
    st.subheader("Output Results")

    st.write(f"Total Data Points : {len(labels)}")
    st.write(f"Number of Clusters : {total_clusters}")
    st.write(f"Noise Points : {noise_points}")
