# Cell 1 (Install necessary libraries)
!pip install tensorflow
!pip install tensorflow-datasets


# Cell 2 (Import Modules)
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
import os


# Cell 3 (Import dataset to colab)
from google.colab import files
uploaded = files.upload()  # To upload your dataset zip file


# Cell 4 (Extracts the dataset zip file)
import zipfile
import os

zip_path = "/content/e_waste_model.zip"  # Change this to your actual ZIP file path
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall("/content")  # Extract to a folder


# Cell 5 (Prepares and processes the dataset for deep learning training)
# Path to the dataset directory
train_dir = '/content/e_waste_model/Train'
val_dir = '/content/e_waste_model/Val'
test_dir = '/content/e_waste_model/Test'

# Initialize ImageDataGenerators with data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values to [0, 1]
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

# Flow data from directories
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),  # Resize images to the required input size for MobileNetV2
    batch_size=32,
    class_mode='categorical'  # Assuming multi-class classification
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)


# Cell 6 (Build and train a transfer learning model using MobileNetV2 as the base)
# Load the pre-trained MobileNetV2 model without the top layers
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base model to prevent training on pre-trained weights
base_model.trainable = False

# Create the model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(train_generator.num_classes, activation='softmax')  # Output layer based on your classes
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator
)


# Cell 7 (Save the model to a file)
model.save('model.h5')

# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model to a file
tflite_model_file = '/content/model.tflite'
with open(tflite_model_file, 'wb') as f:
    f.write(tflite_model)


# Cell 8 (download model file)
from google.colab import files
files.download(tflite_model_file)  # This will download the model.tflite file


