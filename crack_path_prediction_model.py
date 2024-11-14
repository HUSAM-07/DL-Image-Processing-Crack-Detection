import sys
import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input, BatchNormalization, Dropout, Dense, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
import cv2

def check_dependencies():
    required_packages = ['numpy', 'tensorflow', 'sklearn', 'cv2', 'PIL']
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Error: The following required packages are missing: {', '.join(missing_packages)}")
        print("Please install them using pip:")
        print(f"pip install {' '.join(missing_packages)}")
        sys.exit(1)

check_dependencies()

# Constants
INPUT_SHAPE = (224, 224, 1)  # Assuming grayscale images
BATCH_SIZE = 32
EPOCHS = 100
DATASET_FOLDER = "./Dataset"

def load_dataset():
    X, y = [], []
    print(f"Looking for images in: {DATASET_FOLDER}")
    if not os.path.exists(DATASET_FOLDER):
        print(f"Error: Dataset folder '{DATASET_FOLDER}' does not exist.")
        return np.array([]), np.array([])
    
    for filename in os.listdir(DATASET_FOLDER):
        if filename.endswith("_0.png"):
            input_path = os.path.join(DATASET_FOLDER, filename)
            output_path = os.path.join(DATASET_FOLDER, filename.replace("_0.png", "_-1.png"))
            
            if os.path.exists(output_path):
                input_img = load_img(input_path, color_mode="grayscale", target_size=INPUT_SHAPE[:2])
                output_img = load_img(output_path, color_mode="grayscale", target_size=INPUT_SHAPE[:2])
                
                X.append(img_to_array(input_img) / 255.0)
                y.append(img_to_array(output_img) / 255.0)
            else:
                print(f"Missing output image for: {filename}")
    
    print(f"Loaded {len(X)} image pairs")
    return np.array(X), np.array(y)

def extract_features(image):
    # Convert to uint8
    image_uint8 = (image * 255).astype(np.uint8)
    
    # Edge detection
    edges = cv2.Canny(image_uint8, 100, 200)
    
    # Hough transform for line detection
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=10, maxLineGap=5)
    
    if lines is not None:
        # Extract line features (e.g., length, angle)
        line_features = np.array([[np.sqrt((x2-x1)**2 + (y2-y1)**2), np.arctan2(y2-y1, x2-x1)] 
                                  for line in lines for x1, y1, x2, y2 in line])
        return np.mean(line_features, axis=0)
    else:
        return np.zeros(2)

def preprocess_data(X, y):
    X_features = np.array([extract_features(img) for img in X])
    return np.concatenate([X.reshape(X.shape[0], -1), X_features], axis=1), y

def build_model():
    model = Sequential([
        Input(shape=(INPUT_SHAPE[0]*INPUT_SHAPE[1] + 2,)),  # Flattened image + 2 features
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dense(INPUT_SHAPE[0]*INPUT_SHAPE[1], activation='sigmoid'),
        Reshape(INPUT_SHAPE)
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model():
    X, y = load_dataset()
    if len(X) == 0 or len(y) == 0:
        print("No data loaded. Please check the dataset folder and image files.")
        return None
    
    X_processed, y_processed = preprocess_data(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y_processed, test_size=0.2, random_state=42)
    
    model = build_model()
    history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_test, y_test))
    
    model.save('crack_path_prediction_model.h5')
    return history

if __name__ == "__main__":
    history = train_model()
    if history:
        print("Model training completed and saved as 'crack_path_prediction_model.h5'")
    else:
        print("Model training failed. Please check the error messages above.")
