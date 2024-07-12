from keras.utils import to_categorical

# streamlit_app.py

import streamlit as st
import pygame
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


# Initialize pygame
pygame.init()

# Constants
WINDOW_SIZE = (200, 200)
BACKGROUND_COLOR = (0, 0, 0)  # Black
DRAW_COLOR = (255, 255, 255)  # White
IMAGE_SIZE = (28, 28)  # Desired size for all images

# Load dataset
def load_mnist_dataset(train_csv_path, test_csv_path):
    train_data = pd.read_csv(train_csv_path)
    test_data = pd.read_csv(test_csv_path)
    
    # Debug prints to understand dataset structure
    print("Training data shape:", train_data.shape)
    print("Test data shape:", test_data.shape)
    print("Training data columns:", train_data.columns)
    print("Test data columns:", test_data.columns)
    print("First few rows of training data:\n", train_data.head())
    print("First few rows of test data:\n", test_data.head())
    
    X_train = train_data.iloc[:, 1:].values  # pixel values
    y_train = train_data.iloc[:, 0].values  # labels
    X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0  # Normalize pixel values
    y_train = to_categorical(y_train, 10)  # One-hot encode labels
    
    X_test = test_data.values  # pixel values
    X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0  # Normalize pixel values
    
    # Debug prints to verify reshaping
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_test shape:", X_test.shape)
    
    return X_train, y_train, X_test

# Load MNIST dataset
train_csv_path = 'train.csv'
test_csv_path = 'test.csv'
X_train, y_train, X_test = load_mnist_dataset(train_csv_path, test_csv_path)

# Define a simple CNN model
def create_cnn_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Create and train the model
model = create_cnn_model((IMAGE_SIZE[0], IMAGE_SIZE[1], 1), 10)
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

# Function to predict the drawn digit
def predict_digit(image):
    image = image.convert('L').resize(IMAGE_SIZE)
    image = np.array(image) / 255.0
    image = image.reshape(1, IMAGE_SIZE[0], IMAGE_SIZE[1], 1)
    prediction = model.predict(image)
    class_index = np.argmax(prediction)
    return class_index

# Function to run pygame and draw digits
def run_pygame():
    screen = pygame.display.set_mode(WINDOW_SIZE)
    pygame.display.set_caption("Draw a Digit")
    screen.fill(BACKGROUND_COLOR)
    running = True
    drawing = False
    radius = 10
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.image.save(screen, 'digit.png')
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                drawing = True
            elif event.type == pygame.MOUSEBUTTONUP:
                drawing = False
            elif event.type == pygame.MOUSEMOTION:
                if drawing:
                    mouse_pos = pygame.mouse.get_pos()
                    pygame.draw.circle(screen, DRAW_COLOR, mouse_pos, radius)
                    pygame.display.update()

        pygame.display.update()

    pygame.quit()
    return 'digit.png'

# Streamlit UI
st.title("Digit Recognition App")

# Display reference digits
st.header("Reference Digits from Dataset")
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
for i, ax in enumerate(axes.flatten()):
    ax.imshow(X_train[i].reshape(IMAGE_SIZE[0], IMAGE_SIZE[1]), cmap='gray')
    ax.set_title(f'Digit: {np.argmax(y_train[i])}')
    ax.axis('off')
st.pyplot(fig)

# Button to start drawing
if st.button("Start Drawing"):
    with st.spinner("Open pygame window to draw a digit..."):
        image_path = run_pygame()
        if os.path.exists(image_path):
            st.image(image_path, caption="Drawn Digit", use_column_width=True)
            img = Image.open(image_path)
            digit = predict_digit(img)
            st.write(f'Predicted Digit: {digit}')
