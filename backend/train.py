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
import numpy as np
import ssl
from PIL import Image
import urllib.request
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import splitfolders


def main():

    print("Library import done !")
    ####################################################

    data_original = os.listdir('./data/cars dataset')
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

    cars_df.to_csv('./data/transformed/Cars_dataset_final.csv', encoding='utf-8')
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
        #if not os.path.exists(os.path.join('./data/transformed/from',i)):
        if not os.path.exists(os.path.join('./data/transformed/from',i)):
            os.makedirs(os.path.join('./data/transformed/from',i))

    #############################################################

    for c in class_names:
        for i in list(labels[labels['brand']==c]['img_id']):

            get_image = os.path.join('./data/cars dataset',i)       
            if not os.path.exists('./data/transformed/from/'+c+i):
                
                move_image_to_cat = shutil.copy(get_image,'./data/transformed/from/'+c)

    #############################################################

    #### input dataset that want to split
    input_folder = './data/transformed/from'  

    output_folder= './data/transformed/data_splitted'

    splitfolders.ratio(input_folder, output= output_folder, seed=1337, ratio = (0.9, 0, 0.1))

    #############################################################


    class_names = list(cars_df.brand.unique())

    print(sorted(class_names))

    class_names = ['Alfa Romeo', 'Aston Martin', 'Audi', 'Bentley', 'BMW', 'FIAT', 'Ford', 'Honda', 'Hyundai', 'Jaguar', 'Jeep',
    'Kia', 'Land Rover', 'Lexus', 'Maserati', 'Mazda', 'Mercedes-Benz', 'MINI', 'Mitsubishi', 'Nissan', 'Porsche', 'smart', 'Subaru', 'Tesla',
    'Toyota', 'Volkswagen', 'Volvo']

    nb_classes = len(class_names)
    class_names_label = {class_name:i for i, class_name in enumerate(class_names)}
    resizing = (150,150)

    #test here
    (train_images, train_labels), (test_images, test_labels) = img_loading(class_names_label, resizing)
    return train_images, train_labels, test_images, test_labels
    ############################################################
    # CNN function that loads images for the split

def img_loading(class_names_label, resizing):
    split_directory = './data/transformed/data_splitted'
    split_category = ["train", "test"]

    mod_img = []

    for category in split_category:
        path = os.path.join(split_directory, category)
        print(path)
        images = []
        labels = []

        print("Loading {}".format(category))
        counter = 0
        for folder in os.listdir(path):
            if folder != '.DS_Store':
                counter += 1
                print(counter)
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
    print("Fitting images")

    #train_images, train_labels = shuffle(train_images, train_labels, random_state=25)
    train_images = train_images.reshape(-1,150,150,1)
    test_images = test_images.reshape(-1,150,150,1)
    train_images.shape, test_images.shape, train_labels.shape, test_labels.shape

    ###########################################################
def train_model(train_images, train_labels, test_images, test_labels):
    # CNN model building
    #import tensorflow.keras
    from tensorflow.keras import layers
    print("Imported Tensorflow")

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

    cars_fit = cnn_model.fit(train_images, train_labels, batch_size = 128, epochs = 7)

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
    
    #Save the model so we can use it elsewhere
    cnn_model.save('./model/saved_model')
    return


if __name__ == "__main__":
    train_images, train_labels, test_images, test_labels = main()
    train_model(train_images, train_labels, test_images, test_labels)


def brand_id(index):
    brand_name = []
    class_names = ['Alfa Romeo', 'Aston Martin', 'Audi', 'Bentley', 'BMW', 'FIAT', 'Ford', 'Honda', 'Hyundai', 'Jaguar', 'Jeep',
    'Kia', 'Land Rover', 'Lexus', 'Maserati', 'Mazda', 'Mercedes-Benz', 'MINI', 'Mitsubishi', 'Nissan', 'Porsche', 'smart', 'Subaru', 'Tesla',
    'Toyota', 'Volkswagen', 'Volvo']
    brand_name.append(class_names[index])
    print(brand_name)


    #testing with new inputs:
def predict_new_input():
    cnn_model = tf.keras.models.load_model('./model/saved_model')
    cnn_model.summary()
    newimages = []
    #new_input_path = os.path.join("./data/new_input/2017_Fiat_500X_POP_Star_Multiair_1.4_Front.jpg")
    #new_input_path = os.path.join("./data/new_input/Audi_A3_2015_35_17_170_18_4_nan_55_175_24_FWD_4_2_Convertible_QWP.jpg")
    #new_input_path = os.path.join("./data/new_input/WhatsApp Image 2022-05-28 at 18.26.51.jpeg")

    #Create logic to convert image from base64 
    
    # This restores the same behavior as before.
    context = ssl._create_unverified_context()
    link = 'https://images.ctfassets.net/uaddx06iwzdz/5bVowetDTS3KSzwmdf6lkW/15fe2c517e995350b1af2331ca4679e6/vw-golf-7-l-03.jpg'
    req = urllib.request.urlopen(link, context=context)
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    file = cv2.imdecode(arr, -1) 
    new_image = file
    print("opening image")
    resizing = (150,150)

    new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
    new_image = cv2.resize(new_image, resizing)
    newimages.append(new_image)
    newimages = np.array(newimages, dtype = 'float32')
    new_inputs = newimages.reshape(-1,150,150,1)

    # new input into the model
    new_pred = cnn_model.predict(new_inputs)
    new_pred_labels = np.argmax(new_pred, axis=1)

    
    print("PREDICTION FINISHED:")
    index = new_pred_labels[0]
    print(new_pred_labels[0])
    brand_id(index)

    return new_pred_labels





