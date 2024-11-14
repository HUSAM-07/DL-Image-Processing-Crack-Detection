import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Import functions from analysis/main.py
from analysis.main import (
    get_data, extract, process, binarize_input, binarize_output, split, plot, prediction_target_naming, classification_summary, gen_synthetic
)

COLOR_PALETTE = "ch:s=0.00" # "cubehelix"
TARGET = "Merged"
FEATURES = {
    "closest_circle_x": "x",
    "closest_circle_y_abs": r"\vert y \vert",
}

def load_first_frame():
    return pd.read_csv("./first_frame.csv")

def load_last_frame():
    return pd.read_csv("./last_frame.csv")

@st.cache_data(ttl=60*60)
def get_data():
    file_path = "./field_outputs.csv"
    if not os.path.exists(file_path):
        st.error(f"File not found: {file_path}")
        return pd.DataFrame()  # Return an empty DataFrame or handle the error as needed
    return pd.read_csv(file_path)

def main():
    st.set_page_config(layout="wide")

    df = get_data()
    if df.empty:
        st.stop()  # Stop execution if the data is not loaded
    
    df = df.pipe(extract)
    
    # Load the first frame
    first_frame = load_first_frame()
    first_frame = first_frame.pipe(process)
    
    # Load the last frame for comparison
    last_frame = load_last_frame()
    last_frame = last_frame.pipe(process)
    
    st.title("Predict Crack Path")
    
    df = df.pipe(process)
    df[TARGET] = binarize_input(df[TARGET], 0)
    
    df_train, df_test = df.pipe(split)
    X_train = df_train[FEATURES.keys()]
    y_train = df_train[TARGET]
    
    models = [
        (LogisticRegression, dict()), # fit_intercept=True, penalty="l1", solver="liblinear", C=0.01
        (KNeighborsClassifier, dict(n_neighbors=5)),
        (RandomForestClassifier, dict()),
        (HistGradientBoostingClassifier, dict()),
        (MLPClassifier, dict()),
        (GaussianNB, dict()),
        # (SVC, dict(probability=True)),
    ]
    with st.sidebar:
        models_selected = st.multiselect(
            "Models",
            models,
            format_func = lambda x : x[0].__name__,
            default = models[0]
        )
    if len(models_selected)==0:
        models_selected = models
    models_selected_names = [
        model_params[0].__name__
        for model_params
        in models_selected
    ]
    
    with st.spinner("Training"):
        for model, kwargs in models_selected:
            model_name = model.__name__

            model = model()
            if "n_jobs" in dir(model):
                kwargs["n_jobs"] = -1
            model.set_params(**kwargs)
                       
            model.fit(X=X_train, y=y_train)
            
            first_frame[prediction_target_naming(model_name)] = binarize_output(model.predict_proba(first_frame[FEATURES.keys()])[:, 1], 0.4, 0.6)
    
    st.write("First Frame Predictions")
    st.dataframe(first_frame)
    
    st.write("Last Frame for Comparison")
    st.dataframe(last_frame)
    
    # Plot the predictions
    plot(first_frame, prediction_target_naming(models_selected_names[0]), "Plotly")
    plot(last_frame, TARGET, "Plotly")
    
if __name__ == "__main__":
    main()
