import numpy as np
import tensorflow as tf
import cv2

# Load the trained model
model = tf.keras.models.load_model("models/vgg16_dog_cat_classifier.h5")

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict(image_path):
    img = preprocess_image(image_path)
    prediction = model.predict(img)
    return "Dog" if prediction[0][0] > 0.5 else "Cat"

# test oredict
image_path = "dataset/test/other/images.jpeg"
print(f"Predict result: {predict(image_path)}")