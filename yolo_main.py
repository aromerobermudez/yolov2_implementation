
# coding: utf-8

# In[1]:


import os
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

import tensorflow as tf
from keras import backend as K
from keras.layers import Reshape, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.merge import concatenate
from keras.models import load_model, Model


from my_utils import read_classes, read_anchors, preprocess_image, draw_boxes, scale_boxes, WeightReader, yolo_head, yolo_eval
get_ipython().run_line_magic('matplotlib', 'inline')


# ### coco_classes.txt: contains 80 classes
# ### yolo_anchors.txt: the standard 5 anchors used in yolo9000
# ### yolov2.weights: https://pjreddie.com/media/files/yolo.weights   (name the file yolov2.weights)

# In[2]:


class_names = read_classes("model_data/coco_classes.txt")
anchors = read_anchors("model_data/yolo_anchors.txt")
wt_path          = '../yolov2.weights' # name of file containing weights


# ### (IMAGE_H, IMAGE_W) are for the internal calculation. Not the actual image size.
# ### Since we use 32 filters, this means that the grid size is
# #### (IMAGE_H,IMAGE_W) / F = (19x19)
# #### we can reduce (IMAGE_H, IMAGE_W) to (416,416), another standard case of yolo, in this case the internal grid is (13x13)

# In[3]:


IMAGE_H, IMAGE_W = 608, 608 
BOX              = 5 # parameter used to choose filters in the last conv layer: F = BOX * (4 + 1 + CLASS)
CLASS            = len(class_names)
MAX_BOX_N        = 10 # maximum number of boxes per image


# In[4]:


# used in the orgnization layer 
def space_to_depth_x2(x):
    return tf.space_to_depth(x, block_size=2)


# In[5]:


input_image = Input(shape=(IMAGE_H, IMAGE_W, 3))
#true_boxes  = Input(shape=(1, 1, 1, MAX_BOX_N , 4))

# Layer 1
x = Conv2D(32, (3,3), strides=(1,1), padding='same', name='Conv_1', use_bias=False)(input_image)
x = BatchNormalization(name='Batch_norm_1')(x)
x = LeakyReLU(alpha=0.1)(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

# Layer 2
x = Conv2D(64, (3,3), strides=(1,1), padding='same', name='Conv_2', use_bias=False)(x)
x = BatchNormalization(name='Batch_norm_2')(x)
x = LeakyReLU(alpha=0.1)(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

# Layer 3
x = Conv2D(128, (3,3), strides=(1,1), padding='same', name='Conv_3', use_bias=False)(x)
x = BatchNormalization(name='Batch_norm_3')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 4
x = Conv2D(64, (1,1), strides=(1,1), padding='same', name='Conv_4', use_bias=False)(x)
x = BatchNormalization(name='Batch_norm_4')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 5
x = Conv2D(128, (3,3), strides=(1,1), padding='same', name='Conv_5', use_bias=False)(x)
x = BatchNormalization(name='Batch_norm_5')(x)
x = LeakyReLU(alpha=0.1)(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

# Layer 6
x = Conv2D(256, (3,3), strides=(1,1), padding='same', name='Conv_6', use_bias=False)(x)
x = BatchNormalization(name='Batch_norm_6')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 7
x = Conv2D(128, (1,1), strides=(1,1), padding='same', name='Conv_7', use_bias=False)(x)
x = BatchNormalization(name='Batch_norm_7')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 8
x = Conv2D(256, (3,3), strides=(1,1), padding='same', name='Conv_8', use_bias=False)(x)
x = BatchNormalization(name='Batch_norm_8')(x)
x = LeakyReLU(alpha=0.1)(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

# Layer 9
x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='Conv_9', use_bias=False)(x)
x = BatchNormalization(name='Batch_norm_9')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 10
x = Conv2D(256, (1,1), strides=(1,1), padding='same', name='Conv_10', use_bias=False)(x)
x = BatchNormalization(name='Batch_norm_10')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 11
x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='Conv_11', use_bias=False)(x)
x = BatchNormalization(name='Batch_norm_11')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 12
x = Conv2D(256, (1,1), strides=(1,1), padding='same', name='Conv_12', use_bias=False)(x)
x = BatchNormalization(name='Batch_norm_12')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 13
x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='Conv_13', use_bias=False)(x)
x = BatchNormalization(name='Batch_norm_13')(x)
x = LeakyReLU(alpha=0.1)(x)

skip_connection = x

x = MaxPooling2D(pool_size=(2, 2))(x)

# Layer 14
x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='Conv_14', use_bias=False)(x)
x = BatchNormalization(name='Batch_norm_14')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 15
x = Conv2D(512, (1,1), strides=(1,1), padding='same', name='Conv_15', use_bias=False)(x)
x = BatchNormalization(name='Batch_norm_15')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 16
x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='Conv_16', use_bias=False)(x)
x = BatchNormalization(name='Batch_norm_16')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 17
x = Conv2D(512, (1,1), strides=(1,1), padding='same', name='Conv_17', use_bias=False)(x)
x = BatchNormalization(name='Batch_norm_17')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 18
x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='Conv_18', use_bias=False)(x)
x = BatchNormalization(name='Batch_norm_18')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 19
x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='Conv_19', use_bias=False)(x)
x = BatchNormalization(name='Batch_norm_19')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 20
x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='Conv_20', use_bias=False)(x)
x = BatchNormalization(name='Batch_norm_20')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 21
skip_connection = Conv2D(64, (1,1), strides=(1,1), padding='same', name='Conv_21', use_bias=False)(skip_connection)
skip_connection = BatchNormalization(name='Batch_norm_21')(skip_connection)
skip_connection = LeakyReLU(alpha=0.1)(skip_connection)
skip_connection = Lambda(space_to_depth_x2)(skip_connection)

x = concatenate([skip_connection, x])

# Layer 22
x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='Conv_22', use_bias=False)(x)
x = BatchNormalization(name='Batch_norm_22')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 23
x = Conv2D(BOX * (4 + 1 + CLASS), (1,1), strides=(1,1), padding='same', name='Conv_23')(x)

#model = Model([input_image, true_boxes], outputs = x)
model = Model(input_image, outputs = x)


# In[6]:


model.summary()


# ### Load weights

# In[7]:


weight_reader = WeightReader(wt_path)
weight_reader.reset()
nb_conv = 23

for i in range(1, nb_conv+1):
    conv_layer = model.get_layer('Conv_' + str(i))
    
    if i < nb_conv:
        norm_layer = model.get_layer('Batch_norm_' + str(i))
        
        size = np.prod(norm_layer.get_weights()[0].shape)

        beta  = weight_reader.read_bytes(size)
        gamma = weight_reader.read_bytes(size)
        mean  = weight_reader.read_bytes(size)
        var   = weight_reader.read_bytes(size)

        weights = norm_layer.set_weights([gamma, beta, mean, var])       
        
    if len(conv_layer.get_weights()) > 1:
        bias   = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[1].shape))
        kernel = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
        kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
        kernel = kernel.transpose([2,3,1,0])
        conv_layer.set_weights([kernel, bias])
    else:
        kernel = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
        kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
        kernel = kernel.transpose([2,3,1,0])
        conv_layer.set_weights([kernel])


# ### Checking it runs

# In[8]:


image_file = "trafico2.jpg"
image, input_image = preprocess_image("images/" + image_file, model_image_size = (IMAGE_H, IMAGE_W))
netout = model.predict(input_image)
netout.shape


# ### Session and prediction command

# In[9]:


sess = K.get_session()


# In[10]:


def predict(sess, image_file):
    """
    Runs the graph stored in "sess" to predict boxes for "image_file". Prints and plots the preditions.
    
    Arguments:
    sess -- your tensorflow/Keras session containing the YOLO graph
    image_file -- name of an image stored in the "images" folder.
    
    Returns:
    out_scores -- tensor of shape (None, ), scores of the predicted boxes
    out_boxes -- tensor of shape (None, 4), coordinates of the predicted boxes
    out_classes -- tensor of shape (None, ), class index of the predicted boxes
    
    Note: "None" actually represents the number of predicted boxes, it varies between 0 and max_boxes. 
    """

    # Preprocess your image
    image, image_data = preprocess_image("images/" + image_file, model_image_size = (IMAGE_H, IMAGE_W))
    out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes], feed_dict={model.input:image_data , K.learning_phase(): 0})
   
    # Print predictions info
    print('Found {} boxes for {}'.format(len(out_boxes), image_file))
    # Draw bounding boxes on image file
    draw_boxes(image, out_scores, out_boxes, out_classes, class_names)
    # Save the predicted objects
    image.save(os.path.join("out", image_file), quality=90)
    # Display results
    output_image = scipy.misc.imread(os.path.join("out", image_file))
    plt.rcParams["figure.figsize"] = [12,8]
    imshow(output_image)
    
    return out_scores, out_boxes, out_classes


# ### Example with traffic that works

# In[11]:


image_file = "trafico3.jpg"
image_shape = (720.,1280.)#(576.,768.)# (Height , Width)
yolo_outputs = yolo_head(model.output, anchors, len(class_names))
scores, boxes, classes = yolo_eval(yolo_outputs, image_shape, MAX_BOX_N)


# In[12]:


out_scores, out_boxes, out_classes = predict(sess, image_file)


# ### Example with animals that works

# In[13]:


image_file = "animals.jpg"
image_shape = (1920.,1280.)# (Height , Width)
yolo_outputs = yolo_head(model.output, anchors, len(class_names))
scores, boxes, classes = yolo_eval(yolo_outputs, image_shape,MAX_BOX_N)


# In[14]:


out_scores, out_boxes, out_classes = predict(sess, image_file)


# ### Example with traffic that doesn't work (motorcycles are unconventional)

# In[15]:


image_file = "trafico4.jpg"
image_shape = (720.,1280.)#(576.,768.)# (Height , Width)
yolo_outputs = yolo_head(model.output, anchors, len(class_names))
scores, boxes, classes = yolo_eval(yolo_outputs, image_shape,MAX_BOX_N)


# In[16]:


out_scores, out_boxes, out_classes = predict(sess, image_file)

