import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import matplotlib.pyplot as plt

# Load the trained model
model = load_model('weights.keras')

# Mapping of class indices to disease names
label_names = {0: 'Covid-19', 1: 'Normal', 2: 'Viral Pneumonia', 3: 'Bacterial Pneumonia'}

# Function to open the file dialog and select an image
def select_image():
    file_path = filedialog.askopenfilename(title="Select an X-ray Image", 
                                           filetypes=(("Image files", "*.jpg;*.jpeg;*.png"), ("All files", "*.*")))
    if file_path:
        predict_image(file_path)

# Function to preprocess and predict the selected image
def predict_image(file_path):
    # Load the selected image
    img = cv2.imread(file_path)
    img_resized = cv2.resize(img, (128, 128))
    
    # Preprocess the image
    img_normalized = img_resized / 255.0  # Normalize image
    img_array = np.expand_dims(img_normalized, axis=0)  # Add batch dimension
    img_array = np.array(img_array)

    # Predict using the model
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]  # Get class with highest probability

    # Show the image and the predicted class
    display_image_and_prediction(img_resized, predicted_class)

# Function to display the image and the predicted label
def display_image_and_prediction(img, predicted_class):
    # Create a figure
    plt.figure(figsize=(5, 5))

    # Show the image
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f"Predicted: {label_names[predicted_class]}")
    plt.axis('off')
    
    # Show the image
    plt.show()

# Create the main GUI window
root = tk.Tk()
root.title("Respiratory Disease Classification")

# Set window size
root.geometry("400x200")

# Add a button to choose an image and make prediction
select_button = tk.Button(root, text="Choose an X-ray Image", width=30, height=2, command=select_image)
select_button.pack(pady=50)

# Start the GUI loop
root.mainloop()
