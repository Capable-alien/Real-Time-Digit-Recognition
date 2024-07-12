# Real-Time Digit Recognition Application

## Overview
This project is a real-time digit recognition application that allows users to draw a digit on a Pygame window and receive a prediction of the digit using a Convolutional Neural Network (CNN) model. The application leverages machine learning techniques for digit classification and provides a user-friendly interface using Streamlit.

## Usage
1. Install the required dependencies listed in `requirements.txt` using pip:
    `pip install -r requirements.txt`
   
2. Run the Streamlit application using the following command:
    `streamlit run streamlit_app.py`

3. Now, you can access the application in your web browser at the provided URL.

4. Click on the `Start Drawing` button to open the Pygame window and draw a digit. Once you finish drawing, close the pygame window for frame capture and to save it as `digit.png`.

5. Finally, the application will predict the digit from the saved image and display the result.

   
## Project Structure
- `streamlit_app.py`: Streamlit web application for real-time digit recognition.
- `train.csv` and `test.csv`: MNIST dataset CSV files for training and testing the model.
- `requirements.txt`: List of Python dependencies required by the project.

## Features
- Real-time digit recognition from user-drawn images
- User-friendly web interface using Streamlit
- Integration with Pygame for drawing digits
- CNN model for accurate digit classification

## Technical Details
- The digit recognition model is built using Keras with TensorFlow backend.
- Data preprocessing and normalization are performed on the MNIST dataset.
- The application integrates Pygame for a drawing interface and Streamlit for a web interface.
- The CNN model consists of convolutional layers, pooling layers, and dense layers for accurate digit classification.
