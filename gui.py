import tkinter as tk
from tkinter import filedialog
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image, ImageTk

#  Load the trained model
model = tf.keras.models.load_model("models/vgg16_dog_cat_classifier.h5")


# Preprocess images
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"cannot read the file: {image_path}")
        return None
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img


# predict
def predict(image_path):
    img = preprocess_image(image_path)
    if img is None:
        return "Cannot read file"
    prediction = model.predict(img)
    return "🐶 Dog" if prediction[0][0] > 0.5 else "🐱 Cat"


# upload file
def upload_image():
    global img_label, result_label

    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    if not file_path:
        return

    # show picture
    img = Image.open(file_path)
    img.thumbnail((250, 250))
    img = ImageTk.PhotoImage(img)

    img_label.config(image=img)
    img_label.image = img

    # predict result
    result = predict(file_path)
    result_label.config(text=f"Predict Result: {result}", font=("Arial", 14, "bold"))


# create GUI
root = tk.Tk()
root.title("Cat or Dog")

# uoplad button
upload_btn = tk.Button(root, text="Upload picture", command=upload_image, font=("Arial", 12), bg="lightblue")
upload_btn.pack(pady=10)

# show picture
img_label = tk.Label(root)
img_label.pack(pady=10)

# show predict result
result_label = tk.Label(root, text="Predict Result: ", font=("Arial", 12))
result_label.pack(pady=10)

# run GUI
root.mainloop()
