# train.py
import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from skimage import transform, util
import argparse  # For command-line arguments

# 1. Argument Parser Setup
parser = argparse.ArgumentParser(description="Train an arrow image classifier.")
parser.add_argument("--data_dir", type=str, default="pics", help="Path to the directory containing 'up' and 'down' subdirectories.")
parser.add_argument("--output_model", type=str, default="arrow_model.pkl", help="Path to save the trained model.")
args = parser.parse_args()

# 2. Data Loading and Feature Extraction
def load_data(data_dir):
    """Loads images, extracts features, and creates labels."""
    images = []
    labels = []
    for label in ["up", "down"]:
        folder_path = os.path.join(data_dir, label)
        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
            if img is None:
                print(f"Error loading image: {img_path}")
                continue

            # Resize 
            img = cv2.resize(img, (64, 64))

            # Flatten the image into a feature vector
            features = img.flatten()
            images.append(features)
            labels.append(0 if label == "up" else 1)  # 0 for up, 1 for down
    return np.array(images), np.array(labels)


# 3. Data Augmentation
def augment_data(images, labels, num_augmentations=3):
    """Applies data augmentation to the images."""
    augmented_images = []
    augmented_labels = []
    for i in range(len(images)):
        img = images[i].reshape(64, 64)  # Reshape to original image size
        label = labels[i]
        augmented_images.append(images[i])  # Add original image
        augmented_labels.append(label)

        for _ in range(num_augmentations):
            # Random rotation
            angle = np.random.uniform(-15, 15)
            rotated_img = transform.rotate(img, angle, mode='reflect')
            augmented_images.append(rotated_img.flatten())
            augmented_labels.append(label)

            # Random noise
            noisy_img = util.random_noise(img, mode='gaussian', var=0.01)
            augmented_images.append(noisy_img.flatten())
            augmented_labels.append(label)
    return np.array(augmented_images), np.array(augmented_labels)


# 4. Model Training and Evaluation
def train_and_evaluate_model(images, labels):
    """Trains a logistic regression model and evaluates its performance."""
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    model = LogisticRegression(solver='liblinear', random_state=42)  # Example: Logistic Regression
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    return model

# 5. Model Saving
def save_model(model, filepath):
    """Saves the trained model to a file."""
    import pickle
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {filepath}")


# 6. Main Script
if __name__ == "__main__":
    # Use command-line arguments instead of hardcoding the directory
    data_dir = args.data_dir #"pics"  # Path to your "pics" directory
    output_model_path = args.output_model  # "arrow_model.pkl" # Where to save the trained model

    images, labels = load_data(data_dir)
    augmented_images, augmented_labels = augment_data(images, labels, num_augmentations=3)

    model = train_and_evaluate_model(augmented_images, augmented_labels)

    # Save the trained model
    save_model(model, output_model_path)

    print("Training complete.")