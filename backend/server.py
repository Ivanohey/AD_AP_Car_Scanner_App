#Backend server
import pandas as pd
import os
import shutil
import numpy as np
import re
import seaborn as sns
import cv2
import random
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report
from sklearn.utils import shuffle
#from tqdm import tqdm


from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

print("Library import done !")
####################################################

data_original = os.listdir('/Users/nicco/Desktop/ADA_project/cars dataset')
image_df = pd.DataFrame(data_original,columns=['Image'])
sel = image_df[['Image']]
image_df = pd.concat([image_df, sel],axis=1)
image_df.columns = ['Copy', 'Image']
image_df
image_df['check'] = np.where(image_df['Image'] == image_df['Copy'], True, False)
image_df

cars_df = image_df['Copy'].str.split("_", n=16, expand = True)
cars_df = pd.concat([cars_df, image_df], axis=1)

cars_df.drop(cars_df.iloc[:, 3:15], inplace = True, axis = 1)
cars_df.drop(16, inplace = True, axis = 1)
cars_df.drop('Copy', inplace = True, axis = 1)
cars_df.drop('check', inplace = True, axis = 1)
cars_df.columns = ['brand', 'model', 'year', 'car_type', 'img_id']

removed_brands = [".DS","Acura", "Dodge", "Lincoln", "Genesis", 'Buick', 'Cadillac', 'Chevrolet', 'Chrysler', 'Ferrari', 'GMC',
'INFINITI', 'Lamborghini', 'McLaren', 'Rolls-Royce']

removed_car_types = ["Pickup", "3dr", "nan", "Van", "Station Wagon"]

for a in removed_brands:

    cars_df.drop(cars_df[cars_df['brand'] == a].index, inplace = True)

for b in removed_car_types:
    cars_df.drop(cars_df[cars_df['car_type'] == b].index, inplace = True)

cars_df['new_id'] = cars_df['brand']+"_"+cars_df['model']+"_"+cars_df['year']+"_"+cars_df['car_type']+".jpg" #New variable containing the new filenames for pictures
cars_df.reset_index(drop=True)

cars_df

cars_df.to_csv('Cars_dataset_final.csv', encoding='utf-8')
print(cars_df)


#############################################################

sns.set(rc={'figure.figsize':(30,10)})
sns.set(font_scale = 0.6)
sns.countplot(x='brand',data=cars_df)

#############################################################

sns.set(rc={'figure.figsize':(30,10)})
sns.set(font_scale = 0.6)
sns.countplot(x='car_type',data=cars_df)

#############################################################

brands_count = pd.DataFrame(cars_df['brand'].value_counts())
model_counts = pd.DataFrame(cars_df['model'].value_counts())

#############################################################


cars_df.drop(cars_df[cars_df['brand'] == 'import pandas as pd.py'].index, inplace = True)
cars_df.drop(cars_df[cars_df['brand'] == '.vscode'].index, inplace = True)

brands = cars_df.brand.unique()

labels = cars_df.sort_values('brand')

class_names = list(cars_df.brand.unique())

for i in class_names:
    os.makedirs(os.path.join('/Users/nicco/Desktop/ADA_project/test',i))

#############################################################

for c in class_names:
    for i in list(labels[labels['brand']==c]['img_id']):

        get_image = os.path.join('/Users/nicco/Desktop/ADA_project/cars dataset',i)       
        if not os.path.exists('/Users/nicco/Desktop/ADA_project/test/'+c+i):
            
            move_image_to_cat = shutil.copy(get_image,'/Users/nicco/Desktop/ADA_project/test/'+c)

#############################################################

# Here it is necessary to install splitfolder with pip 

import splitfolders

#### input dataset that want to split
input_folder = '/Users/nicco/Desktop/ADA_project/test'  

output_folder= '/Users/nicco/Desktop/ADA_project/data_splitted'

splitfolders.ratio(input_folder, output= output_folder, seed=1337, ratio = (0.8, 0, 0.2))

#############################################################


class_names = list(cars_df.brand.unique())

print(sorted(class_names))

class_names = ['Alfa Romeo', 'Aston Martin', 'Audi', 'Bentley', 'BMW', 'FIAT', 'Ford', 'Honda', 'Hyundai', 'Jaguar', 'Jeep',
'Kia', 'Land Rover', 'Lexus', 'Maserati', 'Mazda', 'Mercedes-Benz', 'MINI', 'Mitsubishi', 'Nissan', 'Porsche', 'smart', 'Subaru', 'Tesla',
'Toyota', 'Volkswagen', 'Volvo']

nb_classes = len(class_names)

class_names_label = {class_name:i for i, class_name in enumerate(class_names)}

resizing = (150,150)

############################################################

# CNN function that loads images for the split

def img_loading():
    split_directory = '/Users/nicco/Desktop/ADA_project/data_splitted'
    split_category = ["train", "test"]

    mod_img = []

    
    for category in split_category:
        path = os.path.join(split_directory, category)
        print(path)
        images = []
        labels = []

        print("Loading {}".format(category))

        for folder in os.listdir(path):
            if folder != '.DS_Store':
                label = class_names_label[folder]
                
                for file in os.listdir(os.path.join(path, folder)):

                    img_path = os.path.join(os.path.join(path, folder), file)
                    image = cv2.imread(img_path)
                    image = cv2.resize(image, resizing)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                    images.append(image)
                    labels.append(label)
                    
        images = np.array(images, dtype = 'float32')
        labels = np.array(labels, dtype = 'int32')
            
        mod_img.append((images, labels))

    return mod_img

##########################################################

(train_images, train_labels), (test_images, test_labels) = img_loading()

##########################################################

# Sample of a modeled image just before cnn process 

plt.imshow(test_images[15])
plt.figure(figsize=(10,10))
for i in range(16):
    plt.subplot(4,4,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    plt.xlabel(class_names[train_labels[i]])
plt.show()

###########################################################

# Fits the image shape with the number of channels for the CNN

#train_images, train_labels = shuffle(train_images, train_labels, random_state=25)
train_images = train_images.reshape(-1,150,150,1)
test_images = test_images.reshape(-1,150,150,1)
train_images.shape, test_images.shape, train_labels.shape, test_labels.shape

###########################################################

# CNN model building

from tensorflow.keras import layers

cnn_model = tf.keras.Sequential([
  tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='valid',input_shape = (150, 150, 1)),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='valid'),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation = tf.nn.relu),
  tf.keras.layers.Dense(27, activation = tf.nn.softmax)
])


# Establishes further parameters such as optimizer, metrics etc...

cnn_model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# cnn model summary

cnn_model.summary()

##########################################################

# CNN model fitting, we throw our train images into the model: COMPUTATION TIME on macbook air no M1 chiped: ~24mins

cars_fit = cnn_model.fit(train_images, train_labels, batch_size = 128, epochs = 5)

##########################################################

test_loss = cnn_model.evaluate(test_images, test_labels)

#########################################################
pred = cnn_model.predict(test_images)
pred_labels = np.argmax(pred, axis=1)
#########################################################

# Final summary of the model trained and tested.

brands = ['Alfa Romeo', 'Aston Martin', 'Audi', 'Bentley', 'BMW', 'FIAT', 'Ford', 'Honda', 'Hyundai', 'Jaguar', 'Jeep', 'Kia',
'Land Rover', 'Lexus', 'Maserati', 'Mazda', 'Mercedes-Benz', 'MINI', 'Mitsubishi', 'Nissan', 'Porsche', 'smart', 'Subaru', 'Tesla', 'Toyota',
'Volkswagen', 'Volvo']

print(classification_report(test_labels, pred_labels, target_names = brands))


########################################################


### FOR IVAN ### Tried to design new

#Define a switch that will take the index of class_names and return the car brand in string.
#!!!ATTENTION!!! je n'ai pas réussi à l'appliquer pour le ndarray obtenu dans le model

#class_names est la list contenant les marques de voitures dans l'ordre alphabétique.
#Le CNN classifie les voitures de 0 à 26, class names reprend cet index et y attribue une marque correspondante en ordre alphabétique/numérique): 0= Alpha Romeo -> 26= Volvo


def brand_id(x):

    brand_name = []

    brand_name.append(class_names[x])

    print(brand_name)


#testing with new inputs:

newimages = []

new_input_path = os.path.join("/Users/nicco/Desktop/ADA_project/new_input/2017_Fiat_500X_POP_Star_Multiair_1.4_Front.jpg")

#new input resize and reshape:
new_image = cv2.imread(new_input_path)
new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
new_image = cv2.resize(new_image, resizing)
newimages.append(new_image)
newimages = np.array(newimages, dtype = 'float32')
new_inputs = newimages.reshape(-1,150,150,1)

# new input into the model
new_pred = cnn_model.predict(new_inputs)
new_pred_labels = np.argmax(new_pred, axis=1)

print(new_pred_labels)

#print(brand_id(new_pred_labels))





