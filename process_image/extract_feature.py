import numpy as np
import os, sys, getopt
import settings

# Code from http://www.marekrei.com/blog/transforming-images-to-feature-vectors/
 
# Main path to your caffe installation
caffe_root = settings.CAFFE_ROOT
 
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
net = caffe.Net(model_prototxt, model_trained, caffe.TEST)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_mean('data', np.load(mean_path).mean(1).mean(1))
transformer.set_transpose('data', (2,0,1))
transformer.set_raw_scale('data', 255.0)
transformer.set_channel_swap('data', (2,1,0)) # RGB -> BGR

data_blob_shape = net.blobs['data'].data.shape
data_blob_shape = list(data_blob_shape)
#net.blobs['data'].reshape(batchsize, data_blob_shape[1], data_blob_shape[2], data_blob_shape[3])

"""
caffe.Classifier(model_prototxt, model_trained,
                       mean=np.load(mean_path).mean(1).mean(1),
                       channel_swap=(2,1,0),
                       raw_scale=255,
                       image_dims=(256, 256))
"""
                       
                       
 
def extract_feature(list_image_file):
    batchsize = len(list_image_file)
    print list_image_file
    net.blobs['data'].reshape(batchsize, data_blob_shape[1], data_blob_shape[2], data_blob_shape[3])
    net.blobs['data'].data[...] = map(lambda x: transformer.preprocess('data',caffe.io.load_image(x)), list_image_file)
        
    #prediction = net.predict(input, oversample=False)
    res = net.forward()
    # predict = np.argmax(res['prob'], axis=1)
    # print predict
    return np.array( net.blobs[layer_name].data.reshape( len(net.blobs[layer_name].data), -1 ) )
