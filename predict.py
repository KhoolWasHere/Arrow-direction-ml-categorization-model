import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt # For visualization

# Assuming you have your load_model function
def load_model(filepath):
    """Loads a trained model from a file."""
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    return model

# --- Main visualization script ---
if __name__ == "__main__":
    model_path = "arrow_model.pkl" # This should be the path to your trained model

    try:
        model = load_model(model_path)

        # Get the coefficients (weights) from the trained model
        # For a binary classification with LogisticRegression, model.coef_
        # will typically be of shape (1, num_features)
        # where num_features is 64 * 64 = 4096 in your case.
        coefficients = model.coef_[0] # Take the first (and only) row for binary classification

        # Reshape the coefficients back into a 2D image format (64x64)
        # This allows us to visualize them like an image
        weight_image = coefficients.reshape(64, 64)

        # Visualize the weight image
        plt.figure(figsize=(8, 6))
        # 'RdBu' colormap is good because it shows positive (red) and negative (blue) values
        # 'cmap="RdBu"' maps higher (positive) coefficients to one color and lower (negative)
        # coefficients to another, with zero in the middle.
        # 'interpolation="nearest"' makes the pixels sharp, not blurry.
        plt.imshow(weight_image, cmap='RdBu', interpolation='nearest')
        plt.colorbar(label='Coefficient Value')
        plt.title('Learned Weights (Pixel Importance) for Up vs. Down')
        plt.xlabel('X-coordinate of Pixel')
        plt.ylabel('Y-coordinate of Pixel')
        plt.show()

        # You can also print some statistics about the weights
        print(f"Min coefficient: {coefficients.min():.4f}")
        print(f"Max coefficient: {coefficients.max():.4f}")
        print(f"Mean coefficient: {coefficients.mean():.4f}")

    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}. Make sure you've trained the model first!")
    except Exception as e:
        print(f"An error occurred: {e}")