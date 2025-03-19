import matplotlib.pyplot as plt
import pickle

# Load training history
with open("models/training_history.pkl", "rb") as f:
    history = pickle.load(f)

# Plot training results
plt.plot(history["accuracy"], label="Train Accuracy")
plt.plot(history["val_accuracy"], label="Val Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()