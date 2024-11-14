# This script does the following:
# Uses the existing folder structure for first and last frame images.
# Extracts features from both first and last frame images using VGG16.
# Creates a dataset of input-output pairs (first frame features to last frame features).
# Trains a simple neural network to predict last frame features from first frame features.
# Allows uploading a new first frame image to predict its corresponding last frame features.

import os
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape
from sklearn.model_selection import train_test_split

# Define constants
INPUT_SHAPE = (224, 224, 3)
FEATURE_SHAPE = (7, 7, 512)  # VGG16 output shape for 224x224 input
DATASET_FOLDER = "./Thahir_Crack_Defect_Interaction_Dataset"

def load_and_preprocess_image(img_path):
    img = load_img(img_path, target_size=INPUT_SHAPE[:2])
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

def extract_features(img_path, feature_extractor):
    img_data = load_and_preprocess_image(img_path)
    features = feature_extractor.predict(img_data)
    return features.flatten()

def create_dataset(dataset_folder, feature_extractor):
    X, y = [], []
    print(f"Searching for images in: {os.path.abspath(dataset_folder)}")
    for filename in os.listdir(dataset_folder):
        if filename.endswith("_0.png"):  # First frame
            first_frame_path = os.path.join(dataset_folder, filename)
            last_frame_path = os.path.join(dataset_folder, filename.replace("_0.png", "_-1.png"))
            
            print(f"Found first frame: {first_frame_path}")
            if os.path.exists(last_frame_path):
                print(f"Found matching last frame: {last_frame_path}")
                X.append(extract_features(first_frame_path, feature_extractor))
                y.append(extract_features(last_frame_path, feature_extractor))
            else:
                print(f"No matching last frame for: {filename}")
    
    if len(X) == 0:
        raise ValueError(f"No valid image pairs found in {dataset_folder}. Please check your image folder.")
    
    return np.array(X), np.array(y)

def create_model(input_shape, output_shape):
    model = Sequential([
        Dense(1024, activation='relu', input_shape=(input_shape,)),
        Dense(512, activation='relu'),
        Dense(np.prod(output_shape), activation='linear'),
        Reshape(output_shape)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')  # Change 'mse' to 'mean_squared_error'
    return model

def main():
    # Load pre-trained VGG16 model
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=INPUT_SHAPE)
    feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)

    # Create dataset
    print("Creating dataset...")
    try:
        X, y = create_dataset(DATASET_FOLDER, feature_extractor)
        print(f"Dataset created with {len(X)} samples.")
    except ValueError as e:
        print(str(e))
        return

    if len(X) == 0:
        print("No valid image pairs found. Please check your image folder.")
        return

    print(f"Dataset created with {len(X)} samples.")

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train model
    input_shape = X.shape[1]
    model = create_model(input_shape, FEATURE_SHAPE)

    print("Training the model...")
    model.fit(X_train, y_train.reshape(-1, *FEATURE_SHAPE), epochs=50, batch_size=32, validation_split=0.2, verbose=1)

    # Evaluate model
    test_loss = model.evaluate(X_test, y_test.reshape(-1, *FEATURE_SHAPE))
    print(f"Test Loss: {test_loss}")

    # Save the model
    model.save('crack_path_prediction_model.h5')
    print("Model saved as 'crack_path_prediction_model.h5'")

if __name__ == "__main__":
    main()
