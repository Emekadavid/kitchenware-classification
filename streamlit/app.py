import streamlit as st
import os 
import numpy as np
import tflite_runtime.interpreter as tflite
from PIL import Image

def preprocess_input(x):
    x /= 127.5
    x -= 1.
    return x

def inference(X_input):
    # we get the model
    interpreter = tflite.Interpreter(model_path='../kitchenware_model.tflite')
    # then the weights
    interpreter.allocate_tensors()

    # get the details about the input and outputs
    input_index = interpreter.get_input_details()[0]['index']
    output_index = interpreter.get_output_details()[0]['index']

    # we can now set the input, do inferenceds and get the predictions
    interpreter.set_tensor(input_index, X_input)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)
    classes = ['cup', 'fork', 'glass', 'knife', 'plate', 'spoon']
    # getting the highest confidence value
    highest_label = classes[np.argmax(preds, 1)[0]]
    return highest_label


st.write("<h1 class='text-center'>The Kitchenware Classification App</h1>", unsafe_allow_html=True)
# get our header file
filepath = os.path.join("..", "images", "kitchenware_image_full.jpg")
st.image(filepath)
st.write("The kitchenware classification app takes an image of a kitchenware item that falls into one of the following 6 classes: cups, glasses, plates, spoons, forks or knives.")
st.write("You have to upload the image and then press the submit button. The app will then make a prediction on the images you uploaded.")
st.write("I downloaded some images from Google. You can use the images in the Google drive by clicking the link below. Just download a file from the folder and upload it by clicking the Browse files button. Or if you choose, you can search for an image that falls into any of the six classes and use your own image.")
st.markdown("[Google Drive Folder with Downloaded images](https://drive.google.com/drive/folders/1VoKciD_Ksre5G1IrWNZHvG2UlfJiAwMQ?usp=share_link)")
st.write("The prediction is after the displayed image.")

img_load = st.file_uploader("Upload a file. Must be jpg or jpeg", type=["jpg", "jpeg"])
if img_load:
    st.image(img_load, width=150, caption="The image you uploaded")
    with Image.open(img_load) as img:
        img = img.resize((299, 299), Image.NEAREST)
    # preprocess the image
    x = np.array(img, dtype='float32') # removed datatype argument here.
    X = np.array([x])
    X = preprocess_input(X)
    result = inference(X)
    st.subheader(f"The model predicted a {result}")


