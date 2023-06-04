

import pandas as pd
import numpy as np
import keras
from keras.utils import load_img
from keras.utils import img_to_array 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
import seaborn as sn
import random
import cv2
import glob
import os
import re
import pandas as pd
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
import plotly.express as px
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, multilabel_confusion_matrix
import tensorflow as tf
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization, Conv2D, MaxPooling2D
from keras.models import Model, Sequential
from keras import regularizers, optimizers
from keras.optimizers import Adam
from keras.layers import concatenate
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from matplotlib import image

def load_data(path_to_data):

  df_train = pd.read_csv(f'{path_to_data}/train.csv')

  # Display the head records from the dataset.
  print(df_train.head())

  # Display the basic statistics for each feature such as count, mean, std, min, max, … etc.
  print(df_train.describe())

  # Display label distribution.
  sns.displot(df_train, x="AdoptionSpeed",discrete=True)

  # Visualize the correlation between each feature and the label
  corrMatrix = df_train.corr()[['AdoptionSpeed']]
  sn.heatmap(corrMatrix, annot=True)
  plt.show()
 
  # Visualize the distribution for all continuous value features using histograms.
  df_train.hist(column=['Age','Fee','VideoAmt','PhotoAmt','Quantity'])
  return df_train

#Splits the dataset into train and test sets following 80/20 partition
def split_data(data):
  y= data.AdoptionSpeed
  x=data.drop('AdoptionSpeed',axis=1)
  train_X, test_X, train_y, test_y = train_test_split(x,y,test_size=0.2)
  return train_X ,test_X,train_y,test_y

# rootdir = '/content/drive/MyDrive/sample_images/train_images1' 
def duplicated_images(rootdir,train_path):
  regex_img = re.compile('[a-zA-Z0-9]*-[1]\.jpg')
  for root, dirs, files in os.walk(rootdir):
    for file in files:
      if regex_img.match(file):
        continue
      else:
        os.remove(rootdir+'/'+file)
      
  #delete rows(pets) from csv that doesn't have images 
  df=pd.read_csv(train_path)
  i=df[df['PhotoAmt'] == 0].index
  df=df.drop(i)
  df.to_csv('train.csv', index=False)

#preprocessing the images
 

def preprocess_image(train_X, test_X, path_img):
    train_images = []
    test_images = []
   
    for img_name in train_X['PetID']:
        img_path = f"{path_img}{img_name}"
        image = tf.keras.preprocessing.image.load_img(img_path)
        image = image.resize((32,32))
        image2 = image.convert('RGB')
        image2 = np.asarray(image2)
        train_images.append(image2)
    
    for test_img_name in test_X['PetID']:
        test_img_path = f"{path_img}{test_img_name}"
        image3 = tf.keras.preprocessing.image.load_img(test_img_path)
        image3 = image3.resize((32,32))
        image3 = image3.convert('RGB')
        image3 = np.asarray(image3)
        test_images.append(image3)
       
    return np.array(train_images), np.array(test_images)

def preprocess_data(train_X,test_X,train_y,test_y):
    pr_train_X = train_X.copy()
    pr_test_X=test_X.copy()
    onehot_encoded = pd.get_dummies(pr_train_X[['Type','Breed1', 'Breed2', 'Color1', 'Color2','Color3','Gender','MaturitySize','FurLength','Vaccinated','Dewormed','Sterilized','Health','State']])
    pr_train_X = pd.concat([pr_train_X, onehot_encoded], axis=1)
    pr_train_X = pr_train_X.drop(['Type','Breed1', 'Breed2', 'Color1', 'Color2','Color3','Gender','MaturitySize','FurLength','Vaccinated','Dewormed','Sterilized','Health','State'],axis=1)
    
    onehot_encoded = pd.get_dummies(pr_test_X[['Type','Breed1', 'Breed2', 'Color1', 'Color2','Color3','Gender','MaturitySize','FurLength','Vaccinated','Dewormed','Sterilized','Health','State']])
    pr_test_X = pd.concat([pr_test_X, onehot_encoded], axis=1)
    pr_test_X = pr_test_X.drop(['Type','Breed1', 'Breed2', 'Color1', 'Color2','Color3','Gender','MaturitySize','FurLength','Vaccinated','Dewormed','Sterilized','Health','State'],axis=1)

    # Scale continuous data to be in the range of [0, 1]
    features = ['Age','Quantity','Fee','VideoAmt','PhotoAmt']
    scaler = MinMaxScaler()
    pr_train_X[features] = scaler.fit_transform(pr_train_X[features])
    pr_test_X[features] = scaler.transform(pr_test_X[features])

    # One-hot encode the labels
    pr_train_y = pd.get_dummies(train_y)
    pr_test_y = pd.get_dummies(test_y)
    return pr_train_X, pr_test_X, pr_train_y, pr_test_y

def create_mlp(dims):
    model = Sequential()
    model.add(Dense(512, input_dim=dims, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(4, activation="softmax"))
  
    return model

def create_CNN(width, height, depth, filters=(16, 32, 64)):
  model = Sequential()
  filters=(16, 32, 64)
  #number of filters = 16, filter size = 3 * 3
  model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(height, width, depth), strides=(1, 1), padding="same"))
  model.add(MaxPooling2D((2, 2)))

  #number of filters = 32, filter size = 3 * 3
  model.add(Conv2D(32, (3, 3), activation='relu'))
  model.add(MaxPooling2D((2, 2)))

  #number of filters = 64, filter size = 3 * 3
  model.add(Conv2D(64, (3, 3), activation='relu'))
  model.add(MaxPooling2D((2, 2)))

  # creating the dense layer or the Fully connected layer FC 
  model.add(Flatten())
  # model.add(Dense(64, activation='relu'))
  # model.add(Dense(filters))
  cnn_model.add(Dense(128, activation='relu'))
  cnn_model.add(Dense(5, activation='softmax'))
 

  return model

def combine_mlp_cnn(mlp_model, cnn_model):
  combinedInput = concatenate([mlp_model.output, cnn_model.output])
  x = Dense(4, activation="relu")(combinedInput)
  x = Dense(5, activation="softmax")(x)  
  model = Model(inputs=[mlp_model.input, cnn_model.input], outputs=x)
  return model

def train_model(train_X, train_y, model):
    # Compile the model with Adam optimizer and an appropriate loss function
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    
    # Train the model on the input training data with a validation set
    model.fit((train_X), train_y, epochs=10, batch_size=32, validation_split=0.2)
    
    # Return the trained model
    return model

def evaluate_model(test_X, test_y, model):

  # predicting the test input 
  y_predict=model.predict(test_X)
  y=np.argmax(y_predict, axis=1)
  test_y=np.argmax(test_y, axis=1)

  # accuracy: (tp + tn) / (p + n)
  accuracy = accuracy_score(test_y, y)

  # precision tp / (tp + fp)
  precision = precision_score(test_y, y, average='macro')
 
  # recall: tp / (tp + fn)
  recall = recall_score(test_y, y, average='macro')
 
  # f1: 2 tp / (2 tp + fp + fn)
  f1_score = 2 * ((recall * precision) / (recall + precision)) 

  return accuracy, precision, recall, f1_score



def main():

  # This is the main method of your script and performs the following tasks:
  # 1. Load the pet adoption dataset
  path_img = '/content/drive/MyDrive/sample_images/train_images1/'  
  train_path='/content/train.csv'
  duplicated_images(path_img,train_path)
  path_to_data='/content'
  df=load_data(path_to_data)

  df["PetID"]=df["PetID"].apply(lambda PetID: PetID+'-1'+".jpg")

  # 2. Split the data into train and testing sets
  train_X,test_X,train_y,test_y = split_data(df)
  # 3. Resize the images to be 32 × 32 pixels
  train_imgs,test_imgs = preprocess_image(train_X, test_X, path_img)
  # 4. Preprocess the categorical and numerical data as the follows:
  # - Normalize the continuous data
  # - Encode the categorical data as one-hot representation
  # - Encode the labels as one-hot representation
  pr_train_X, pr_test_X, pr_train_y, pr_test_y = preprocess_data(train_X,test_X,train_y,test_y)
  pr_train_X.drop('PetID', axis=1, inplace=True)
  pr_train_X.drop('Name', axis=1, inplace=True)
  pr_train_X.drop('RescuerID', axis=1, inplace=True)
  pr_train_X.drop('Description', axis=1, inplace=True)

  pr_test_X.drop('PetID', axis=1, inplace=True)
  pr_test_X.drop('Name', axis=1, inplace=True)
  pr_test_X.drop('RescuerID', axis=1, inplace=True)
  pr_test_X.drop('Description', axis=1, inplace=True)


  # 5. Define the following two models:
  # - MLP model
  # dims=12
  dims=pr_train_X.shape[1]
  mlp_model = create_mlp(dims)

  # - CNN model
  # cnn_model = create_CNN(32, 32, 3, filters= 64)
  cnn_model=create_CNN(32, 32, 3, filters=(16, 32, 64)) #G

  # 6. Combine the MLP and CNN models into one model
  model = combine_mlp_cnn(mlp_model, cnn_model)

  # 7. Train the combined model

  pr_train_y = np.array(pr_train_y).astype(np.float32)
  pr_train_X = np.array(pr_train_X).astype(np.float32)
  # train_X = np.concatenate([pr_train_X, train_imgs], axis=1)
  trained_model=train_model([pr_train_X, train_imgs], np.array(pr_train_y), model)

  # 8. Evaluate the trained model based on its accuracy, precision, recall, and f1-score
  test_y = np.array(pr_test_y).astype(np.float32)
  test_X = np.array(pr_test_X).astype(np.float32)
  accuracy, precision, recall, f1_score = evaluate_model([test_X, test_imgs], test_y, trained_model)

  print('Accuracy: ',accuracy)
  print('Precision: ',precision)
  print('Recall: ',recall)
  print('F1_score: ',f1_score)
main()