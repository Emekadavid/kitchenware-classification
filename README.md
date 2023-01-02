# Kitchenware Classification

This is a project that is organized by [Datatalks.Club](https://datatalks.club/). In this competition, one has to train a deep learning model in tensorflow or pytorch to classify kitchenware items. I used tensorflow and keras for this task. The kitchenware items are:
- cups
- glasses
- plates
- spoons
- forks
- knives

As an image classification model, when given the image of one of the above-listed kitchenware items, the model will output probailities for each of the six classes. The highest probability serves as the model's final classification. 

## Image Dataset

You can get the dataset to reproduce the classification on your own system from the [Kaggle Competition page](https://www.kaggle.com/competitions/kitchenware-classification/overview). I would recommend using a GPU for this task because a CPU would take hours if not days. I used the free GPU from [Saturn Cloud](https://saturncloud.io/). Saturn Cloud gives 30 hours of free GPU every month. 

## Exploratory Data Analysis

Since deep learning models don't need extensive data analysis and exploration, I did only minimal EDA. It was discovered that in the dataset, there were 6 kitchenware items or classes as outlined above. The train dataset was divided into 80 percent final train and 20 percent validation. Which means there were 4447 images in the final train dataseet and 1112 images in the validation dataset. The test dataset had 3808 images. 

Most of the images had distracting items at the borders, that was why data augmentation was necessary. You would notice that I carried out extensive data augmentation in order to make the model not overfit due to the distracting features in the images. 

##  Model Training

The model was trained on tensorflow with a pre-trained convolutional network (CNN) model, Xception, with two convolutional layers. Then a hidden layer gave good results using relu activation. The output layer was trained using softmax. 

You can find all this information on the `notebook-kitcheware.ipynb` file in the repo.

## Python Script

TBC