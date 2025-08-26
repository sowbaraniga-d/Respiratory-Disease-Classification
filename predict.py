import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import tkinter as tk
from tkinter import filedialog

# Load the trained model
model = tf.keras.models.load_model('pneumonia_classifier_model.h5')

# Labels Translator
label_names = {
    0: 'Covid-19',
    1: 'Normal',
    2: 'Viral Pneumonia',
    3: 'Bacterial Pneumonia'
}

# Function to predict the image
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]

    predicted_label = label_names[predicted_class]
    print(f"The image is classified as: {predicted_label}")
    
    return predicted_label

# GUI to choose image
def choose_image_gui():
    file_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=(("JPEG files", "*.jpg"), ("PNG files", "*.png"), ("All files", "*.*"))
    )

    if file_path:
        predicted_label = predict_image(file_path)
        result_label.config(text=f"Prediction: {predicted_label}")
    else:
        print("No file selected")

# Setup GUI
def setup_gui():
    window = tk.Tk()
    window.title("Respiratory Classifier")

    global result_label
    result_label = tk.Label(window, text="Prediction: ", font=("Arial", 14))
    result_label.pack(pady=20)

    choose_button = tk.Button(window, text="Choose Image", command=choose_image_gui, font=("Arial", 14))
    choose_button.pack(pady=10)

    window.mainloop()

setup_gui()
