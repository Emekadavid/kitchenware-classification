from keras_image_helper import create_preprocessor
import requests
import numpy as np 
from PIL import Image

# with Image.open('./images/fork1.jpg') as img:
#     img = img.resize((299, 299), Image.NEAREST)

# def preprocess_input(x):
#     x /= 127.5
#     x -= 1.
#     return x

# image_array = np.array(img, dtype='float32')
# compressed_array = np.array([image_array])

# X = preprocess_input(compressed_array)
# get the image. You can change the image in line 5 based on your choice
preprocessor = create_preprocessor('xception', target_size=(299,299))
X = preprocessor.from_path('./images/plate.jpg')

kitchenware_data = {"items": X.tolist()}
url = 'http://localhost:9696/kitchenware'
## post the image information in json format 
response = requests.post(url, json=kitchenware_data)
# get the result 
result = response.json()
print("The image is a", result)
