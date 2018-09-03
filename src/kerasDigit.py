from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing import image
from keras.utils import to_categorical



def load_mnist_data():

    (train_images , train_labels) , (test_images , test_labels) = mnist.load_data()
    return train_images , train_labels , test_images , test_labels



trainImg , trainLab , testImg , testLab =load_mnist_data()

def info_mnist_data(trImg=trainImg , trLab=trainLab , teImg= testImg , teLab = testLab):
    classes = np.unique(trLab)
    nClasses = len(classes)
    print "Output Sayisi : {}  \nSiniflar : {}".format(nClasses , classes)

    #sample figure on plot
    print('Training data shape : ', trainImg.shape, trainLab.shape)
    plt.figure(figsize=[10,5])
    plt.subplot(121)
    plt.imshow(trainImg[0 , : , : ] , cmap='gray')
    plt.show(block= False)
    raw_input("Press Enter to exit")
    plt.close()

info_mnist_data(trainImg , trainLab , testImg , testLab)












"""
plt.figure(figsize=[10,5])
plt.subplot(121)
plt.imshow(train_images [ 0 , : , : ] , cmap= 'gray')
plt.show()
"""




