from picamera import PiCamera
from time import sleep
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

# Load the pre-trained Keras model
from keras.models import load_model
model = 'path/to/saved/model'

# Initialize the camera
camera = PiCamera()

# Set camera resolution
camera.resolution = (256,256)  # Adjust as needed to match the model's input size

# Capture image
image_path = "captured_image.jpg"
camera.capture(image_path)

# Close the camera
camera.close()

# Preprocess the image
img = image.load_img(image_path, target_size=(256, 256))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# Perform prediction
predictions = model.predict(x)
decoded_predictions = decode_predictions(predictions, top=3)[0]

# Print the top 3 predictions
for pred in decoded_predictions:
    print(pred[1], ": ", pred[2])

# Wait for a few seconds before exiting 
sleep(2)
