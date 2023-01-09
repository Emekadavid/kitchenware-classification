import tensorflow as tf
#import tensorflow.keras as keras

model = tf.keras.models.load_model("xception_vlarge2_31_0.968.h5")
converter = tf.lite.TFLiteConverter.from_keras_model(model)

tflite_model = converter.convert()

with open('kitchenware_model.tflite', 'wb') as f_out:
    f_out.write(tflite_model)