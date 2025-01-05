import streamlit as st
import tensorflow as tf
import numpy as np
import os
import cv2
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
from PIL import Image
import random

# Constants (matching training configuration)
INPUT_SHAPE = (224, 224, 1)
DATASET_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), "generated_frames")

def load_model():
    """Load the trained model"""
    model_path = 'crack_path_prediction_model.keras'
    if not os.path.exists(model_path):
        st.error(f"Model file '{model_path}' not found! Please train the model first.")
        return None
    return tf.keras.models.load_model(model_path)

def extract_features(image):
    """Extract features from image (matching training process)"""
    image_uint8 = (image * 255).astype(np.uint8)
    edges = cv2.Canny(image_uint8, 100, 200)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=10, maxLineGap=5)
    
    if lines is not None:
        line_features = np.array([[np.sqrt((x2-x1)**2 + (y2-y1)**2), np.arctan2(y2-y1, x2-x1)] 
                                for line in lines for x1, y1, x2, y2 in line])
        return np.mean(line_features, axis=0)
    else:
        return np.zeros(2)

def process_image(image_path):
    """Load and preprocess image for prediction"""
    img = load_img(image_path, color_mode="grayscale", target_size=INPUT_SHAPE[:2])
    img_array = img_to_array(img) / 255.0
    features = extract_features(img_array)
    
    # Combine image and features (matching training process)
    X = np.concatenate([img_array.reshape(-1), features])
    return X, img_array

def predict_crack_propagation(model, input_data):
    """Make prediction using the model"""
    prediction = model.predict(input_data.reshape(1, -1), verbose=0)  # Added verbose=0 to reduce output
    return prediction.reshape(INPUT_SHAPE)

def main():
    st.set_page_config(layout="wide")  # Better layout for image comparison
    st.title("Crack Propagation Prediction")
    
    # Load model
    model = load_model()
    if model is None:
        return
    
    # Get list of first frames
    first_frames = [f for f in os.listdir(DATASET_FOLDER) if f.endswith('first_frame.png')]
    
    if not first_frames:
        st.error("No first frame images found in the dataset folder!")
        return
    
    # Sidebar controls
    with st.sidebar:
        st.header("Controls")
        if st.button("Select Random Image", key="random"):
            selected_image = random.choice(first_frames)
            st.session_state.selected_image = selected_image
        else:
            # Use session state to maintain selection
            if 'selected_image' not in st.session_state:
                st.session_state.selected_image = first_frames[0]
            selected_image = st.session_state.selected_image
        
        # Add a dropdown for manual selection
        selected_image = st.selectbox(
            "Or choose a specific image:",
            first_frames,
            index=first_frames.index(st.session_state.selected_image)
        )
    
    # Main content
    st.subheader(f"Analyzing: {selected_image}")
    
    # Process and display images
    image_path = os.path.join(DATASET_FOLDER, selected_image)
    X, original_img = process_image(image_path)
    
    # Get actual final frame
    final_frame = selected_image.replace('first_frame.png', 'last_frame.png')
    final_frame_path = os.path.join(DATASET_FOLDER, final_frame)
    
    # Make prediction with progress indicator
    with st.spinner('Predicting crack propagation...'):
        prediction = predict_crack_propagation(model, X)
    
    # Create columns for display
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### Initial State")
        st.image(original_img, caption="First Frame", use_container_width=True)
    
    with col2:
        st.markdown("### Predicted")
        st.image(prediction, caption="Predicted Crack Path", use_container_width=True)
    
    with col3:
        st.markdown("### Actual")
        if os.path.exists(final_frame_path):
            actual_img = load_img(final_frame_path, color_mode="grayscale", target_size=INPUT_SHAPE[:2])
            st.image(actual_img, caption="Actual Final State", use_container_width=True)
        else:
            st.error("Actual final frame not available")
    
    # Add metrics/analysis section
    if os.path.exists(final_frame_path):
        st.subheader("Analysis")
        actual_img = load_img(final_frame_path, color_mode="grayscale", target_size=INPUT_SHAPE[:2])
        actual_array = img_to_array(actual_img) / 255.0
        
        # Calculate similarity metrics
        mse = np.mean((prediction - actual_array) ** 2)
        similarity = 1 / (1 + mse)  # Normalized similarity score
        
        # Display metrics
        metrics_col1, metrics_col2 = st.columns(2)
        with metrics_col1:
            st.metric("Prediction Accuracy", f"{similarity*100:.2f}%")
        with metrics_col2:
            st.metric("Mean Squared Error", f"{mse:.4f}")

if __name__ == "__main__":
    main()
