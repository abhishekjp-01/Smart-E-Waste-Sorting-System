# Smart E-Waste Sorting System

This project is a Smart E-Waste Sorting System prorotype that combines machine learning, Raspberry Pi, and Arduino Uno to automatically classify and sort electronic waste into the correct bins.
<img src="assets/Capture.png" alt="Prototype Preview" width="600">

## Project Overview
-	A deep learning model is trained in Google Colab to classify different types of e-waste.
-	The Raspberry Pi processes real-time images of waste using the trained model and sends classification results to the Arduino via serial communication.
-	The Arduino Uno controls the mechanical system (waste bin rotation and waste gate opening) to direct the item into the appropriate bin.

## Repository Contents
-	Model training in Colab → Jupyter/Colab notebook for training the image classification model.
-	Raspberry Pi Code → Python code for running the trained model on Raspberry Pi, capturing images, and sending results to Arduino.
-	Arduino Code → Embedded C++ code for Arduino Uno to control motors/actuators for bin rotation and gate operation.
## Features
-	Automated e-waste classification using deep learning.
-	Integration of Raspberry Pi (for AI inference) and Arduino (for hardware control).
-	Real-time waste sorting into designated bins.
## Hardware & Software Used
-	Raspberry Pi
-	Arduino Uno
-	Servo motor
-	Stepper motor
-	Camera module (for image capture)

