import cv2
import serial
import time
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
current_val = 5
flag = True
# Open serial connection to Arduino (Change port if needed)
arduino = serial.Serial('/dev/serial0', 9600)  
time.sleep(2)  # Wait for Arduino to initialize

# Function to send a command to Arduino
def send_command(predicted_class_index):
    global flag
    if flag == True:  # Ensure a valid class is detected
        flag = False
        
        if (predicted_class_index == 5 or predicted_class_index == -1):
            return
        arduino.write(f"{predicted_class_index}\n".encode())  # Send as string
        time.sleep(0.1)
        print(f"Sent: {predicted_class_index}")
        time.sleep(0.6)  # Small delay
        

# Load the TensorFlow Lite model
model_path = '' # model path here
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Function to preprocess image
def preprocess_image(frame):
    img = cv2.resize(frame, (224, 224))  # Resize to model input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize
    return img_array

# Function to perform inference
def run_inference(frame, threshold=0.8):
    img_array = preprocess_image(frame)  # Preprocess frame
    
    # Get input/output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], img_array)
    
    # Run inference
    interpreter.invoke()
    
    # Get the output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    # Get predicted class index
    predicted_class_index = np.argmax(output_data)
    predicted_class_prob = output_data[0][predicted_class_index]
    
    # If confidence is too low, return -1 (uncertain)
    if predicted_class_prob < threshold:
        return -1
    
    return predicted_class_index

# Smooth predictions over last N frames
history = []
def smooth_prediction(prediction, history_size=10):
    history.append(prediction)
    if len(history) > history_size:
        history.pop(0)
    return np.bincount(history).argmax()  # Return most frequent class

# Define class labels
class_labels = ['Battery', 'LED', 'Mobile', 'Mouse', 'PCB', 'Use_Me']

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    # Perform inference
    predicted_class_index = run_inference(frame)
    time.sleep(1)
    

    if predicted_class_index == -1:
        predicted_class_label = "Uncertain"
    else:
        smoothed_class_index = smooth_prediction(predicted_class_index)
        predicted_class_label = class_labels[smoothed_class_index]

    # Display result on frame
    cv2.putText(frame, predicted_class_label, (55, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_AA)
    cv2.rectangle(frame, (10, 10), (200, 100), (0, 255, 0), 2)
    
    cv2.imshow('Real-Time Object Detection', frame)

    print(f"Predicted: {predicted_class_index}")
    
    if predicted_class_index == 4:
        predicted_class_index = 2
    
    
    if current_val != predicted_class_index:
        flag = True
        current_val = predicted_class_index
        send_command(predicted_class_index)
    # send_command(predicted_class_index)  # Send data to Arduino
   
    print(flag)
    # Break loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
arduino.close()  # Close serial connection at the end
