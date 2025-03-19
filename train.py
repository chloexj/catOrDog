import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pickle
# data path
train_dir = "dataset/train"
val_dir = "dataset/test"

# data enhancement
train_datagen = ImageDataGenerator(rescale=1.0/255.0, rotation_range=20, zoom_range=0.2, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1.0/255.0)

train_generator = train_datagen.flow_from_directory(train_dir, target_size=(224, 224), batch_size=32, class_mode="binary")
val_generator = val_datagen.flow_from_directory(val_dir, target_size=(224, 224), batch_size=32, class_mode="binary")

# Build a VGG16 Transfer Learning Model
base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers:
    layer.trainable = False

x = Flatten()(base_model.output)
x = Dense(512, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=x)
model.compile(optimizer=Adam(learning_rate=0.0001), loss="binary_crossentropy", metrics=["accuracy"])

# Train the Model
history = model.fit(train_generator, validation_data=val_generator, epochs=10, steps_per_epoch=len(train_generator), validation_steps=len(val_generator))

# ensure history is not null
if history.history:
    with open("models/training_history.pkl", "wb") as f:
        pickle.dump(history.history, f)
    print("Training history save successfully！")
else:
    print("Training history is null，didn't save！")

# save the Model
model.save("models/vgg16_dog_cat_classifier.h5")

print("Training complete, model saved!！")
