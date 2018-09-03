from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing import image
from keras.utils import to_categorical


(train_images , train_labels) , (test_images , test_labels) = mnist.load_data()

classes = np.unique(train_labels)
nClasses = len(classes)

print"Number of outputs :  {}  \nOutput Classes : {}  ".format(nClasses , classes)



