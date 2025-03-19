import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("CUDA support:", tf.test.is_built_with_cuda())
print("GPU device:", tf.config.list_physical_devices('GPU'))
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))