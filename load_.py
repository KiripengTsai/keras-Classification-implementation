from keras.models import load_model
import os

from PIL import Image

import numpy as np

model = load_model('model.h5')
fpath = './data/0_38.jpg'
image = Image.open(fpath)
data = np.array(image) / 255.0
data1 = np.array([data])
p = model.predict(data1)
print(p)
