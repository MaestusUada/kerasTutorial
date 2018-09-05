from keras.models import load_model
from PIL import Image
import numpy as np


def testInput(path):
    model = load_model('../models/mnistCNN.h5')

    for index in range(1):
       img = Image.open(path).convert("L")
       img = img.resize((28,28))
       im2arr = np.array(img)
       im2arr = im2arr.reshape(1,28,28,1)
       # Predicting the Test set results
       y_pred = model.predict(im2arr)
       print(y_pred)
       print np.argmax(y_pred , axis=1)
    
