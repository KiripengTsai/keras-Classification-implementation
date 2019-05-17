import os
#图像读取库
from PIL import Image
#矩阵运算库
import numpy as np
import keras
from keras.layers import Dense,Dropout,Flatten
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

data_dir = "data"

def read_data(data_dir):
    datas = []
    labels = []
    fpaths = []
    datab  = []
    for fname in os.listdir(data_dir):
        fpath = os.path.join(data_dir, fname)
        fpaths.append(fpath)
        image = Image.open(fpath)
        data = np.array(image) / 255.0
        label = int(fname.split("_")[0])
        datas.append(data)
        labels.append(label)


    datas = np.array(datas)
    labels = np.array(labels)

    print("shape of datas: {}\tshape of labels: {}".format(datas.shape, labels.shape))
    return fpaths, datas, labels

fpaths, datas, labels = read_data(data_dir)

num_classes = len(set(labels))

datas.astype(float)
encoder = LabelEncoder()
encoded_Y = encoder.fit_transform(labels)
dummy_y = np_utils.to_categorical(encoded_Y)


def baseline_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(32,32,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3))  # 3分类
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', #多分类
              optimizer='rmsprop',
              metrics=['accuracy'])
    return model


model = baseline_model()

model.fit(datas,dummy_y,
          batch_size=32,
          epochs=100)
scores = model.evaluate(datas, dummy_y, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
print(datas.shape)
p = model.predict(datas)
model.save('model.h5')
'''
for i in range(150):
    if p[i][0]>p[i][1] and p[i][0]>p[i][2]:
        print("{}====>1".format(labels[i]+1))
    elif p[i][1]>p[i][0] and p[i][1]>p[i][2]:
        print("{}====>2".format(labels[i]+1))
    else:
        print("{}====>3".format(labels[i]+1))
'''