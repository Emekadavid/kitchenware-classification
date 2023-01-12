import numpy as np
import tflite_runtime.interpreter as tflite
from keras_image_helper import create_preprocessor

preprocessor = create_preprocessor('xception', target_size=(299,299))
X = preprocessor.from_path('./images/plate.jpg')

# we get the model
interpreter = tflite.Interpreter(model_path='kitchenware_model.tflite')
# then the weights
interpreter.allocate_tensors()

# get the details about the input and outputs
input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

# we can now set the input, do inferenceds and get the predictions
interpreter.set_tensor(input_index, X)
interpreter.invoke()
preds = interpreter.get_tensor(output_index)
classes = ['cup', 'fork', 'glass', 'knife', 'plate', 'spoon']
# getting the highest confidence value
highest_label = classes[np.argmax(preds, 1)[0]]
print("The image is a", highest_label)