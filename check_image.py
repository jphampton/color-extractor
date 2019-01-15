import cv2
import numpy as np

from color_extractor import ImageToColor

def make_labels_set(thresh):
    colors = np.array(np.meshgrid(np.arange(256), np.arange(256), np.arange(256))).T.reshape(-1, 3)
    labels = np.array([""]*colors.shape[0],dtype="S10")
    whites = (colors  > 256*(1-thresh)).all(axis=1)
    labels[whites] = 'white'
    labels[~whites] = 'non-white'
    return colors, labels

npz = np.load('color_names.npz')
img_to_color = ImageToColor(npz['samples'], npz['labels'])
print("Samples shape: {}".format(npz['samples'].shape))
print("Labels shape: {}".format(npz['labels'].shape))
print("Labels type: {}".format(npz['labels'].dtype))
print(img_to_color)
img = cv2.imread('/home/james/Downloads/rahul_test_images/testImages/ford.png')
print('PRINTING COLOR')
print(img_to_color.get(img))

print(img[0,0,:])
print(img[0, -1, :])
print(img[-1, 0, :])
print(img[-1, -1, :])