import cv2
import numpy as numpy
import os
from random import shuffle
from tqdm import tqdm
import tensorflow as tf 
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model

MODEL = '/home/john/PycharmProjects/ec-autonav/final_model.h5'

ARROWS = '/home/john/Pictures/Arrow Trainingset/'
NOT = '/home/john/Pictures/Others trainingset2/'

#test_this = '/home/john/Pictures/Arrow Testset'
test_this = '/home/john/Pictures/nn_stuff/test'

train_data = '/home/john/Pictures/nn_stuff/train'
test_data = '/home/john/Pictures/nn_stuff/test'

""" def rename_files():
    i = 0

    for filename in os.listdir(NOT):
        name = NOT + 'not.' + str(i) + '.png'
        src = NOT + filename
        os.rename(src,name)
        i += 1 """

def one_hot_label(img):
    label = img.split('.')[0]
    if label == 'arrow':
        ohl = np.array([1,0])
    elif label == 'not':
        ohl = np.array([0,1])
    return ohl



def label_dataset(direct):
    dataset = []
    for i in tqdm(os.listdir(direct)):
        path = os.path.join(direct,i)
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (34,34))
        dataset.append([np.array(img),one_hot_label(i)])
    shuffle(dataset)
    return dataset

def load_datasets():
    train_images = label_dataset(train_data)
    test_images = label_dataset(train_data)
    return train_images, test_images

    # define cnn model
def define_model():
    model = Sequential()
    model.add(InputLayer(input_shape=[34,34,1]))
    model.add(Conv2D(filters=32,kernel_size=5,strides=1,padding='same',activation='relu'))
    model.add(MaxPool2D(pool_size=5,padding='same'))

    model.add(Conv2D(filters=50,kernel_size=5,strides=1,padding='same',activation='relu'))
    model.add(MaxPool2D(pool_size=5,padding='same'))
    
    model.add(Conv2D(filters=80,kernel_size=5,strides=1,padding='same',activation='relu'))
    model.add(MaxPool2D(pool_size=5,padding='same'))
    
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(2, activation='softmax'))
    optimizer = Adam(lr=1e-3)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def predict(model):
    count = 0
    for i in tqdm(os.listdir(test_this)):
            path = os.path.join(test_this,i)
            #img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
            #img = cv2.resize(img,(64,64))
            #image.reshape(1,64,64,1)
            img = load_img(path,grayscale=True,target_size=(34,34))
            img = img_to_array(img)
            img = img.reshape(1,34,34,1)
            output = model.predict_classes(img)
            if output[0] == 1:
                count += 1
    accuracy = (27-count)/27.0
    print('predictor is ' + str(accuracy) + '% accurate')

training_images, testing_images = load_datasets()

tr_img_data = np.array([i[0] for i in training_images]).reshape(-1,34,34,1)
tr_lbl_data = np.array([i[1] for i in training_images])
tst_img_data = np.array([i[0] for i in testing_images]).reshape(-1,34,34,1)
tst_lbl_data = np.array([i[1] for i in testing_images])

model = define_model()
model.fit(x=tr_img_data,y=tr_lbl_data,epochs=50,batch_size=100)
model.summary()
model.save('final_model.h5')
#model = load_model(MODEL)
predict(model)


# fig = plt.figure(figsize=(14,14))

# for cnt, data in enumerate(testing_images[10:40]):

#     y = fig.add_subplot(6,5,cnt+1)
#     img = data[0]
#     data = img.reshape(1,64,64,1)
#     model_out = model.predict([data])

#     if np.argmax(model_out) == 1:
#         str_label = 'arrow'
#     else:
#         str_label = 'not'
    
#     y.imshow(img, cmap='gray')
#     plt.title(str_label)
#     y.axes.get_xaxis().set_visible(False)
#     y.axes.get_yaxis().set_visible(False)

