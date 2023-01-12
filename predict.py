import numpy as np
import tflite_runtime.interpreter as tflite

from flask import Flask, request, jsonify

def make_prediction(X):
    # change x to float 32
    X = np.float32(X)


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
    return highest_label

app = Flask("kitchenware")
@app.route("/kitchenware", methods=['POST'])
def predict():
    image_data = request.get_json()
    # Make the prediction noting the actual payload
    prediction = make_prediction(image_data['items'])
    return jsonify(str(prediction))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)    