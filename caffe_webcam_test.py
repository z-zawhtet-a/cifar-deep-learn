__author__ = 'zawhtetaung'
import numpy as np
import matplotlib.pyplot as plt
import sys
import caffe
import cv2

# Make sure that caffe is on the python path:
caffe_root = '/Users/zawhtetaung/Documents/caffe-master/examples/'

# Set the right path to your model definition file, pretrained model weights,
MODEL_FILE = '/Users/zawhtetaung/Google Drive/Experiment/Deep Learning/models/bvlc_reference_caffenet/deploy.prototxt'
PRETRAINED = '/Users/zawhtetaung/Google Drive/Experiment/Deep Learning/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'

imagenet_labels_filename = '/Users/zawhtetaung/Documents/caffe-master/Auxillary Data/synset_words.txt'

net = caffe.Classifier(MODEL_FILE, PRETRAINED,
                       mean=np.load('/usr/local/lib/python2.7/site-packages/caffe/imagenet/ilsvrc_2012_mean.npy'),
                       channel_swap=(2, 1, 0),
                       raw_scale=255,
                       image_dims=(256, 256))
net.set_phase_test()
net.set_mode_cpu()

cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)
labels = np.loadtxt(imagenet_labels_filename, str, delimiter='\t')

while True:

    ret, frame = cam.read()

    if ret:


        cv2.imshow("Test", frame[:256, :256])   # Take 256 x 256 crop from webcam

        key = cv2.waitKey(33)

        if key == ord('r'):

            cv2.imwrite('test.jpg', frame[:256, :256])  # Had to save first to file (Why?!)
            input_image = caffe.io.load_image('test.jpg')
            prediction = net.predict([input_image], oversample=True)  # Predict
            top_k = net.blobs['prob'].data[4].flatten().argsort()[-1:-4:-1] # Top 4 results
            print labels[top_k]

        elif key == ord('q'):

            break
