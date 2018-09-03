from keras.models import model_from_json
from matplotlib.pyplot import plot as plt
from keras.preprocessing import image
from keras.layers import Dropout
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot  as plt
import numpy as np

json_file = open('../datasets/model_num.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("../datasets/model_num.h5")
print("Loaded model from disk")

loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

img = image.load_img(path="../datasets/2.png",color_mode="grayscale",target_size=(28,28))
img = image.img_to_array(img)
X_train = img.reshape(1, 28, 28, 1)
X_train = X_train.astype("float32")
img_class = loaded_model.predict(X_train)
print "Class : {} ".format(img_class)