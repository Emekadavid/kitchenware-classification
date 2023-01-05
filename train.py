#Downloading the data from Kaggle
# 
# Follow these steps to download the data from Kaggle. Remove the comments and 
# run it once then you don't need to run the codes again.


# Configuration environment in order to download the data from Kaggle
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow import keras

import os
import zipfile
import kaggle 

from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.xception import preprocess_input

from tensorflow.keras.preprocessing.image import ImageDataGenerator

os.environ['KAGGLE_USERNAME'] = "" # username from the Kaggle json file
os.environ['KAGGLE_KEY'] = "" # key from the Kaggle json file

# download kaggle competition file
competition_name = "kitchenware-classification"
file_name = "kitchenware-classification.zip"
# Download the file
kaggle.api.competition_download_file(competition_name, file_name)


# make directory for unpacking the files 
if not os.path.exists("data"):
    os.mkdir("data")
# unzip the zipped files to the data directory
# Open the zip file in read mode
with zipfile.ZipFile('kitchenware-classification.zip', 'r') as zip_ref:
    # Extract all the contents of zip file in current directory
    zip_ref.extractall()    
os.remove("kitchenware-classification.zip") 


# create the train and validation datasets
df_train_full = pd.read_csv('data/train.csv', dtype={'Id': str})
df_train_full['filename'] = 'data/images/' + df_train_full['Id'] + '.jpg'


val_cutoff = int(len(df_train_full) * 0.8)
df_train = df_train_full[:val_cutoff]
df_val = df_train_full[val_cutoff:]


# the images will now be processed
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_ds = train_datagen.flow_from_dataframe(
    df_train,
    x_col='filename',
    y_col='label',
    target_size=(150, 150),
    batch_size=32,
)

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

val_ds = val_datagen.flow_from_dataframe(
    df_val,
    x_col='filename',
    y_col='label',
    target_size=(150, 150),
    batch_size=32,
)

# Let me show you how the classes are structured in the network. 
print(train_ds.class_indices)
# So, you can see that for the kitchen ware task, we have cups, forks, glasses, knives, 
# plates and spoons. 

## Training with high image resolution
# 
# In the notebook, notebook-kitchenware.ipynb, I used images of size 150x150
# to experiment with hyper parameter tuning. Now I will use larger images 
# of size 299x299 here.

def make_model(input_size= 150, learning_rate=0.01, size_inner=100, droprate=0.5): # size_inner is the number of neurons in the first layer
    base_model = Xception(
        weights='imagenet',
        include_top=False,
        input_shape=(input_size, input_size, 3)
    )

    base_model.trainable = False

    #########################################

    inputs = keras.Input(shape=(input_size, input_size, 3))
    base = base_model(inputs, training=False)
    vectors = keras.layers.GlobalAveragePooling2D()(base)
    
    inner = keras.layers.Dense(size_inner, activation='relu')(vectors)
    drop = keras.layers.Dropout(droprate)(inner)
    
    outputs = keras.layers.Dense(6)(drop)
    
    model = keras.Model(inputs, outputs)
    
    #########################################

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    loss = keras.losses.CategoricalCrossentropy(from_logits=True)

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy']
    )
    
    return model

input_size = 299

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input, width_shift_range=5, height_shift_range=10, 
                                  shear_range=10, zoom_range=0.1, horizontal_flip=True, vertical_flip=True)

train_ds = train_datagen.flow_from_dataframe(
    df_train,
    x_col='filename',
    y_col='label',
    target_size=(input_size, input_size),
    batch_size=32,
)

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

val_ds = val_datagen.flow_from_dataframe(
    df_val,
    x_col='filename',
    y_col='label',
    target_size=(input_size, input_size),
    batch_size=32,
)

checkpoint = keras.callbacks.ModelCheckpoint(
    'xception_vlarge_{epoch:02d}_{val_accuracy:.3f}.h5',
    save_best_only=True,
    monitor='val_accuracy',
    mode='max'
)

learning_rate = 0.001
size = 500
droprate = 0.8

model = make_model(
    input_size=input_size,
    learning_rate=learning_rate,
    size_inner=size,
    droprate=droprate
)

import matplotlib.pyplot as plt 

history = model.fit(train_ds, epochs=50, validation_data=val_ds,
                   callbacks=[checkpoint])

hist = history.history
plt.plot(hist['val_accuracy'], label='val')
plt.plot(hist['accuracy'], label='train')

plt.legend()
plt.show()

# I like the results from the above assessment. 0.96 for validation is a good result. 
# But maybe I can do better. It is evident that the model may be resonating around 
# a local minimum. Let me reduce the learning rate to see if I can get a better result than an oscillation around a constant. Otherwise, I go for the above results. 

checkpoint = keras.callbacks.ModelCheckpoint(
    'xception_vlarge2_{epoch:02d}_{val_accuracy:.3f}.h5',
    save_best_only=True,
    monitor='val_accuracy',
    mode='max'
)

learning_rate = 0.0005
size = 500
droprate = 0.8

model = make_model(
    input_size=input_size,
    learning_rate=learning_rate,
    size_inner=size,
    droprate=droprate
)

history = model.fit(train_ds, epochs=50, validation_data=val_ds,
                   callbacks=[checkpoint])

hist = history.history
plt.plot(hist['val_accuracy'], label='val')
plt.plot(hist['accuracy'], label='train')

plt.legend()
plt.show()

# Reducing the learning rate to 0.0005 gave a better fit this time. You can see that 
# the accuracy on validation data mimics that of train data with very low difference. 
# Therefore, there is little chance of overfitting in this model. I will use this 
# model henceforth. Also, the percentage accuracy is better than the other models. 
# 
# Now, it's time to use the model on test data. 

## Using The Model On Test Data

df_test = pd.read_csv('data/test.csv', dtype={'Id': str})
df_test['filename'] = 'data/images/' + df_test['Id'] + '.jpg'
print(df_test.head())

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)


test_ds = test_datagen.flow_from_dataframe(
    df_test,
    x_col='filename',
    class_mode='input',
    target_size=(299, 299),
    batch_size=32,
    shuffle=False
)

# First import the best model

model = keras.models.load_model('xception_vlarge2_31_0.968.h5')

# Now I will evaluate the model to see its accuracy on unseen test data.

y_pred = model.predict(test_ds)

# Getting the classes of the different images

classes = np.array(list(train_ds.class_indices.keys()))
print(classes)

# Generating the classes for the prediction

predictions = classes[y_pred.argmax(axis=1)]

# Will now create a file for submission to Kaggle

df_submission = pd.DataFrame()
df_submission['filename'] = test_ds.filenames
df_submission['label'] = predictions

df_submission['Id'] = df_submission.filename.str[len('data/images/'):-4]

del df_submission['filename']

df_submission[['Id', 'label']].to_csv('submissionlarge.csv', index=False)

# NB: The final model got a 0.96 accuracy on Kaggle. 


