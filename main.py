import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tkinter as tk
from tkinter import filedialog

# labels Translator 
label_names = {
    0: 'Covid-19',
    1: 'Normal',
    2: 'Viral Pneumonia',
    3: 'Bacterial Pneumonia'
}

# Set paths for your dataset
train_dir = 'Train_Dataset/'  # Replace with your actual train dataset path
test_dir = 'Test/'    # Replace with your actual test dataset path

# Initialize ImageDataGenerators for data augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1.0/255.0)

# Load train and test datasets (using the numeric folder names)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)

# Define the model architecture using MobileNetV2 as the base model
basemodel = MobileNetV2(weights='imagenet', include_top=False, input_tensor=tf.keras.Input(shape=(128, 128, 3)))
basemodel.trainable = False

model = models.Sequential([
    basemodel,
    layers.GlobalAveragePooling2D(),
    layers.Dense(4, activation='softmax')  # 4 classes, softmax for multi-class classification
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=30,
    validation_data=test_generator,
    validation_steps=test_generator.samples // test_generator.batch_size
)

# After training, evaluate the model on the test set and display classification report
test_predictions = model.predict(test_generator)
predicted_classes = np.argmax(test_predictions, axis=1)

# True labels from test generator
true_labels = test_generator.classes

# Print classification report with class names
print(classification_report(true_labels, predicted_classes, target_names=[label_names[i] for i in range(4)]))

# Save the model after training
model.save('pneumonia_classifier_model.h5')

# Prediction on a new image
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make a prediction
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)

    # Map the numeric prediction to a human-readable label
    predicted_label = label_names[predicted_class[0]]
    print(f"The image is classified as: {predicted_label}")
    
    return predicted_label

# GUI to choose image using Tkinter
def choose_image_gui():
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    # Open file dialog to choose an image
    file_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=(("JPEG files", "*.jpg"), ("PNG files", "*.png"), ("All files", "*.*"))
    )

    if file_path:  # If a file was selected
        predicted_label = predict_image(file_path)
        print(f"Predicted Label: {predicted_label}")
        result_label.config(text=f"Prediction: {predicted_label}")

    else:
        print("No file selected")

# Setup the GUI
def setup_gui():
    window = tk.Tk()
    window.title("Respiratory Classifier")

    # Add a label to show the result
    global result_label
    result_label = tk.Label(window, text="Prediction: ", font=("Arial", 14))
    result_label.pack(pady=20)

    # Add a button to choose an image
    choose_button = tk.Button(window, text="Choose Image", command=choose_image_gui, font=("Arial", 14))
    choose_button.pack(pady=10)

    window.mainloop()

# Start the GUI
setup_gui()
