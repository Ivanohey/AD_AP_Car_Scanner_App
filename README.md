# Using Augmented Reality and Deep Learning to recognize cars in the street
## Advanced Programming project
### Ivan Kostine, Niccolo Cherubini, Andrea Ferrazzo

This repository contains all the files for our project of the cours Advanced Programming of the Master in Business Analytics of HEC Lausanne

## Scope of the project

For this project we chose to create a prototype of a mobile application allowing a user to take a photo of a car in the street and to obtain information on its brand and model thanks to a deep learning model running in the backend of the application.

## Installation
Cloning the project:
```
git clone git@github.com:Ivanohey/ADA-AP.git
```
The dataset and model used by our Deep Learning logic are included in the repository. In case you have issues downloading the data, you can find the dataset used in our project here: https://www.kaggle.com/datasets/prondeau/the-car-connection-picture-dataset

To run our project you will need an installation of Python and the following dependencies:

```
tensorflow
fastAPI
kivy
pydantic
numpy
base64
io
PIL
shutil
seaborn
matplotlib
sklearn
splitfolders
cv2
pandas
json
```
## Frontend
After making sure that all the Python dependecies are installed, from the root of the project navigate to the frontend folder and run the frontend file:

````
cd /frontend
python frontend.py
````

## Backend
After making sure that all the Python dependecies are installed, from the root of the project navigate to the backend folder and run the backend file:
The server uses FastAPI and listenes to http://127.0.0.1:8000 when ran locally. 

````
cd /backend
python server.py
````
In some cases, running the backend can fail because of the tensorflow library. If you are not using a computer equiped with a GPU, you might need to consider installing another version of tensorflow (more information can be found here https://www.tensorflow.org/install/pip): 

