import numpy as np
import os, sys, getopt

# Code from http://www.marekrei.com/blog/transforming-images-to-feature-vectors/
 
# Main path to your caffe installation
caffe_root = '/path/to/your/caffe/'
 
# Model prototxt file
model_prototxt = caffe_root + 'models/bvlc_googlenet/deploy.prototxt'
 
# Model caffemodel file
model_trained = caffe_root + 'models/bvlc_googlenet/bvlc_googlenet.caffemodel'
 
# File containing the class labels
imagenet_labels = caffe_root + 'data/ilsvrc12/synset_words.txt'
 
# Path to the mean image (used for input processing)
mean_path = caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy'
 
# Name of the layer we want to extract
layer_name = 'pool5/7x7_s1'
 
sys.path.insert(0, caffe_root + 'python')
import caffe

# Setting this to CPU, but feel free to use GPU if you have CUDA installed
caffe.set_mode_cpu()
# Loading the Caffe model, setting preprocessing parameters
net = caffe.Classifier(model_prototxt, model_trained,
                       mean=np.load(mean_path).mean(1).mean(1),
                       channel_swap=(2,1,0),
                       raw_scale=255,
                       image_dims=(256, 256))
                       
                       
 
def extracr_feature(list_image_file):
    input = []
    for image_path in reader:
        image_path = image_path.strip()
        input.append(caffe.io.load_image(image_path))
        
    prediction = net.predict([input_image], oversample=False)
    return [ x.reshape(1, -1) for x in net.blobs[layer_name].data ]