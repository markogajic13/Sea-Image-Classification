# Sea-Image-Classification

# CNN model

## Overview 
This project focuses on classifying images into two categories: "under sea" and "above sea". The goal is to accurately identify whether an image depicts scenes underwater or above the surface. The classification model is built using Convolutional Neural Networks (CNNs), a deep learning architecture known for its effectiveness in image recognition tasks.

## Dataset
The dataset consists of a collection of images sourced from both underwater and above-sea environments. These images are labeled accordingly to facilitate supervised learning. The dataset is divided into training, validation, and test sets to train, validate, and evaluate the model's performance.

## Model Architecture
The classification model is constructed using TensorFlow and Keras, with a sequential architecture of convolutional and pooling layers followed by fully connected layers. The model is trained using the training dataset and optimized using the Adam optimizer. During training, the model learns to distinguish between underwater and above-sea images by minimizing the categorical cross-entropy loss.

## Evaluation
The trained model's performance is evaluated using the test dataset to assess its accuracy in classifying unseen images. Additionally, the training and validation accuracy and loss are monitored to ensure the model's generalization capability and prevent overfitting.

# TensorFlow Serving attained with Docker

## Overview
This Dockerfile sets up a TensorFlow Serving environment using the official TensorFlow Serving runtime as the parent image. It copies the pre-trained models (versions 1 and 2) along with a configuration file (`tf.config`) into the container.

## Dependencies
- TensorFlow Serving: Official Docker image for TensorFlow Serving runtime

## Configuration
- **Model Copy**: The Dockerfile copies the directories containing the models (`1` and `2`) and the configuration file (`tf.config`) into the appropriate location within the container (`/models/vectrino`).
- **Exposed Ports**: The Dockerfile exposes the gRPC port (8500) and the HTTP port (8501) to allow communication with TensorFlow Serving.
- **Entry Point**: TensorFlow Serving is started as the entry point with specified configurations (`--model_name=vectrino`, `--port=8500`, `--rest_api_port=8501`, and `--model_config_file=/models/vectrino/tf.config`).
- ### Point
- **Building the Docker Image**: Use the Dockerfile to build a Docker image containing the TensorFlow Serving environment and the pre-trained models.
- **Running the Docker Container**: Start a Docker container from the built image, which will host the TensorFlow Serving instance with the specified models and configurations.

# FastAPI backend service

## Overview
This code implements a FastAPI-based web service for image classification using a pre-trained TensorFlow model. It allows users to upload an image, which is then classified into one of two categories: "Ispod mora" (Under the sea) or "Iznad mora" (Above the sea). The classification is based on a deep learning model deployed using TensorFlow Serving.

## Dependencies
- FastAPI: A modern, fast (high-performance), web framework for building APIs with Python 3.7+
- TensorFlow: An open-source machine learning framework for building and training neural networks
- uvicorn: ASGI server implementation, using uvloop and httptools
- numpy: A powerful N-dimensional array object used for numerical computing
- PIL: Python Imaging Library for handling images
- requests: A simple HTTP library for Python

## Functionality
- **/ping Endpoint**: A GET request to this endpoint returns a simple "Hello, I am alive" message, indicating that the service is running.
- **/predict Endpoint**: A POST request to this endpoint expects an image file as input. It reads the uploaded image, preprocesses it, and sends it to a TensorFlow Serving endpoint for prediction. The predicted class and confidence score are then returned as JSON.

## TensorFlow Serving
The code interacts with a TensorFlow Serving endpoint (`endpoint`) to perform inference. This endpoint should be running a TensorFlow Serving instance hosting the pre-trained model named "vectrino".

## CORS Configuration
The service allows cross-origin resource sharing (CORS) from specified origins (`http://localhost` and `http://localhost:3000`) to enable web clients to make requests to the API from these domains.

# Frontend service layer realized with React.js

## Overview
This React component allows users to upload an image for classification. It displays the uploaded image (if available), classifies it using a backend API, and shows the predicted class along with confidence percentage. Users can also clear the uploaded image and classification results.

## Dependencies
- React: JavaScript library for building user interfaces
- Material-UI: React components for faster and easier web development
- axios: Promise-based HTTP client for the browser and Node.js

## Features
- **Image Upload**: Users can upload an image for classification.
- **Classification**: After uploading, the image is sent to a backend API for classification. The predicted class and confidence percentage are displayed.
- **Clear Functionality**: Users can clear the uploaded image and classification results with the "Clear" button.

## Component Structure
- **AppBar**: Displays the application logo.
- **Container**: Contains the main content of the component.
- **Card**: Displays the uploaded image or dropzone for uploading.
- **Table**: Shows the predicted class and confidence percentage.
- **CircularProgress**: Indicates loading state while classifying the image.
- **Button**: Allows users to clear the uploaded image and classification results.



