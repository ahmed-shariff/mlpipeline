import numpy as np
import tensorflow as tf
import math
import itertools

import os
from inspect import getsourcefile
#DEV SECTION: this section, I defined it to help the development of the model script 


#END DEV SECTION
#This section is sepcially needed if the model scripts are not in the same directory from which the pipline is being executed
from utils import add_script_dir_to_PATH
from utils import ExecutionModeKeys
from utils import Versions
from helper import Model
from helper import DataLoader

add_script_dir_to_PATH(os.path.abspath(os.path.dirname(getsourcefile(lambda:0))))

from dataLoader import DataLoader
#import models.dataLoader



class TestingDataLoader(DataLoader):
  def __init__(self):
    print("creating dataloader")

  def get_train_sample_count(self):
    return 1000

  def get_test_sample_count(self):
    return 1000

  def get_train_input_fn(self, **kargs):
    return lambda:print("got input form train input function")

  def get_test_input_fn(self):
    return lambda:print("got input form test input function")
  
  
class TestingModel(Model):
  def __init__(self, versions, **args):
    super().__init__(versions, **args)

  def pre_execution_hook(self, version, model_dir, exec_mode=ExecutionModeKeys.TEST):
    print("Pre execution")
    print("Version spec: ", version)
    self.current_version = version

  def get_current_version(self):
    return self.current_version

  def get_trained_step_count(self):
    return 10

  def train_model(self, input_fn, steps):
    print("steps: ", steps)
    print("calling input fn")
    input_fn()

  def evaluate_model(self, input_fn, steps):
    print("steps: ", steps)
    print("calling input fn")
    input_fn()

dl = TestingDataLoader()
v = Versions(0.01, dl)
v.addV("ha")
MODEL = TestingModel(versions = v)

# MODEL_SUMMERY = ""
# MODEL_SUMMERY_SET = False
# IS_MODEL = True

# USE_ALL_CLASSES = False
# CLASSES_COUNT = 10
# BATCH_SIZE = 21
# EPOC_COUNT = 73
# CLASSES_OFFSET = 0
# ALLOW_DELETE_MODEL_DIR=False
# RESTART_GLOBAL_STEP=False

# _LEARNING_RATE=0.05

# epsilon = 0.01
# decay = 0.9
# k = 5
# MODEL = None
# VERSIONS = Version(
#   learning_rate = _LEARNING_RATE,
#   use_all_classes= USE_ALL_CLASSES,
#   classes_count = CLASSES_COUNT,
#   batch_size = BATCH_SIZE,
#   epoc_count = EPOC_COUNT)
# #model_dir_suffix = "14-class")
# #  classes_offset = CLASSES_OFFSET, #using default value

# versionName = "6-classes"
# VERSIONS.addV(versionName, model_dir_suffix = versionName, classes_offset=["pizza", "samosa", "spring_rolls", "chocolate_cake", "cup_cakes", "donuts"])
# VERSIONS.getVersion(versionName).block_size = [1,2,4,3]
# VERSIONS.getVersion(versionName).factorize=[False,True,True]

# def add_summery(addition):
#   global MODEL_SUMMERY
#   if not MODEL_SUMMERY_SET:
#     MODEL_SUMMERY += "\t-" + addition + "\n"

# def model_summery_finalize():
#   global MODEL_SUMMERY_SET
#   MODEL_SUMMERY_SET = True

# def bn_wrapper(layer, mode):
#   return tf.nn.relu(tf.layers.batch_normalization(layer, momentum = 0.9,
#                                                   training = mode==tf.estimator.ModeKeys.TRAIN))

# def res_blocks(starting_layer, block_count, level, initial_filters, mode,
#                block_size = 2, kernal_size = 3, factorize=False):
#   conv = starting_layer
#   for block in range(block_count):
#     parent_conv = conv
#     filter_count = initial_filters * (2**level)
#     for l in range(block_size):
#       if not factorize:
#         add_summery("conv-{0}-{1}-{2}: size:{3}, kernal:{4}, filters: {5}".format(level,block+1,l+1,
#                                                                                 256/2**(level + 1),
#                                                                                 kernal_size,
#                                                                                 filter_count))
#         conv = tf.layers.conv2d(conv,
#                                 filters= filter_count,
#                                 kernel_size = kernal_size,
#                                 padding = "SAME",
#                                 activation = None,
#                                 use_bias = False,
#                                 kernel_initializer=tf.random_normal_initializer(),
#                                 name = "conv-{0}-{1}-{2}".format(level,
#                                                                block,
#                                                                l))
#         conv = bn_wrapper(conv, mode)
#       else:
#         add_summery("conv-{0}-{1}-{2}-0: size:{3}, kernal:{4}, filters: {5}".format(level,block+1,l+1,
#                                                                                 256/2**(level + 1),
#                                                                                 [kernal_size,1],
#                                                                                 filter_count))
#         conv = tf.layers.conv2d(conv,
#                                 filters= filter_count,
#                                 kernel_size = [kernal_size,1],
#                                 padding = "SAME",
#                                 activation = None,
#                                 use_bias = False,
#                                 kernel_initializer=tf.random_normal_initializer(),
#                                 name = "conv-{0}-{1}-{2}-0".format(level,
#                                                                block,
#                                                                l))
#         conv = bn_wrapper(conv, mode)

#         add_summery("conv-{0}-{1}-{2}-1: size:{3}, kernal:{4}, filters: {5}".format(level,block+1,l+1,
#                                                                                 256/2**(level + 1),
#                                                                                 [1,kernal_size],
#                                                                                 filter_count))
#         conv = tf.layers.conv2d(conv,
#                                 filters= filter_count,
#                                 kernel_size = [1,kernal_size],
#                                 padding = "SAME",
#                                 activation = None,
#                                 use_bias = False,
#                                 kernel_initializer=tf.random_normal_initializer(),
#                                 name = "conv-{0}-{1}-{2}-1".format(level,
#                                                                block,
#                                                                l))
#         conv = bn_wrapper(conv, mode)
        
#     conv = tf.nn.relu(conv + parent_conv)
#     add_summery("adding and relu")
#     if factorize:
#       add_summery("\tblock:{0} layers".format(block_size*2))
#     else:
#       add_summery("\tblock:{0} layers".format(block_size))

#   return conv
# # def bn_wrapper_(inputs, mode):

# #   is_training = mode == tf.estimator.ModeKeys.TRAIN
# #   scale = tf.Variable(tf.ones(inputs.get_shape()[1:]))
# #   beta = tf.Variable(tf.zeros(inputs.get_shape()[1:]))
# #   pop_mean = tf.Variable(tf.zeros(inputs.get_shape()[1:]), trainable=False)
# #   pop_var = tf.Variable(tf.ones(inputs.get_shape()[1:]), trainable=False)
# #   if is_training:
# #     batch_mean, batch_var = tf.nn.moments(inputs,[0])
# #     train_mean = tf.assign(pop_mean,
# #                            pop_mean * decay + batch_mean * (1 - decay))
# #     train_var = tf.assign(pop_var,
# #                           pop_var * decay + batch_var * (1 - decay))
# #     with tf.control_dependencies([train_mean, train_var]):
# #       return tf.nn.batch_normalization(inputs,
# #                                        batch_mean, batch_var, beta, scale, epsilon)
# #   else:
# #     return tf.nn.batch_normalization(inputs,
# #                                      pop_mean, pop_var, beta, scale, epsilon)
  
# def get_model_fn(version_name, depth, reset_model_dir = False):
#   """
# versions: reserved for future
# depth: number of classes. This is added to allow easier testing
# The function returns a function that can be pased as the model_fn in the estimator
#   """
#   version = VERSIONS.getVersion(version_name)
#   _LEARNING_RATE = version.learning_rate
#   filters = 32
#   depth=101

#   block_sizes = version.block_size
#   b_1 = block_sizes[0]
#   b_2 = block_sizes[1]
#   b_3 = block_sizes[2]
#   b_4 = block_sizes[3]

#   factorize=version.factorize
#   b_2_fac = factorize[0]
#   b_3_fac = factorize[1]
#   b_4_fac = factorize[2]
  

#   block_size = 2#version.block_size
#   kernal_size = 3#version.kernal_size

#   add_summery("Learning rate: {0}".format(_LEARNING_RATE))
#   add_summery("All layers: batch_normalized and relu")
  
#   def model_fn(features, labels, mode):
#     """Model function for CNN."""

#     if isinstance(features, dict):
#       features = features["f"]
    
#     # Input Layer
#     # Reshape X to 4-D tensor: [batch_size, width, height, channels]
#     # MNIST images are 28x28 pixels, and have one color channel
#     #add_summery("Input layer: [512, 512, 3]")
#     input_layer = tf.reshape(features, [-1, 256, 256, 3],"input_layer")
#     #labels = tf.Print(labels, [labels], "The Label::")
#     # Convolutional Layer #1
#     # Computes 32 features using a 5x5 filter with ReLU activation.
#     # Padding is added to preserve width and height.
#     # Input Tensor Shape: [batch_size, 28, 28, 1]
#     # Output Tensor Shape: [batch_size, 28, 28, 32]

#     # i: [-1, 256, 256, 3]
#     # o: [-1, 128, 128, filtersize]
#     level = 0
#     add_summery("conv-{0}-0-0: size:{1}, kernal:{2}, filters: {3}, strides: 1".
#                 format(level,
#                        256/2,
#                        3,
#                        filters * (2**level)))
    
#     conv0 = tf.layers.conv2d(input_layer,
#                              filters=filters * (2**level),
#                              kernel_size = 3,
#                              padding = "SAME",
# 			     strides = 1,
#                              activation = None,
#                              kernel_initializer=tf.random_normal_initializer(),
#                              use_bias=False, name = "conv0")
#     conv0 = bn_wrapper(conv0, mode)

#     conv0 = tf.layers.max_pooling2d(conv0, 3,2,"same")

    
#     level = 1
#     # i: [-1, 128,128, fliterSize1]
#     # o: [-1, 64, 64, filtersize2]
#     add_summery("conv-{0}-0-0: size:{1}, kernal:{2}, filters: {3}, strides: 1".
#                 format(level,
#                        256/2**(level),
#                        3,
#                        filters * (2**(level-1))))
    
#     conv1 = tf.layers.conv2d(conv0,
#                              filters= filters * (2**(level-1)),
#                              kernel_size = 3,
#                              padding = "SAME",
# 			     strides = 1,
#                              activation = None,
#                              kernel_initializer=tf.random_normal_initializer(),
#                              use_bias=False, name = "conv1-1")
#     conv1 = bn_wrapper(conv1, mode)

#     add_summery("conv-{0}-0-0: size:{1}, kernal:{2}, filters: {3}, strides: 1".
#                 format(level,
#                        256/2**(level),
#                        3,
#                        filters * (2**(level-1))))
#     conv1 = tf.layers.conv2d(conv1,
#                              filters= filters * (2**(level-1)),
#                              kernel_size = 3,
#                              padding = "SAME",
# 			     strides = 1,
#                              activation = None,
#                              kernel_initializer=tf.random_normal_initializer(),
#                              use_bias=False, name = "conv1-2")
#     conv1 = bn_wrapper(conv1, mode)

#     add_summery("conv-{0}-0-0: size:{1}, kernal:{2}, filters: {3}, strides: 1".
#                 format(level,
#                        256/2**(level),
#                        3,
#                        filters * (2**level)))

#     conv1 = tf.layers.conv2d(conv1,
#                              filters= filters * (2**level),
#                              kernel_size = 3,
#                              padding = "SAME",
# 			     strides = 1,
#                              activation = None,
#                              kernel_initializer=tf.random_normal_initializer(),
#                              use_bias=False, name = "conv1-3")
#     conv1 = bn_wrapper(conv1, mode)

#     add_summery("maxpool: [3,3] kernal, strides: 2")
#     conv1 = tf.layers.max_pooling2d(conv1,3,2,"same")
    
#     conv1 = res_blocks(conv1, b_1, level, filters, mode,
#                        block_size = block_size, kernal_size=kernal_size)

    
#     level = 2
#     # i: [-1, 64,64, fliterSize1]
#     # o: [-1, 32, 32, filtersize2]
#     add_summery("conv-{0}-0-0: size:{1}, kernal:{2}, filters: {3}, strides: 1".
#                 format(level,
#                        256/2**(level),
#                        3,
#                        filters * (2**level)))
#     conv2 = tf.layers.conv2d(conv1,
#                              filters= filters * (2**level),
#                              kernel_size = 3,
#                              padding = "SAME",
# 			     strides = 1,
#                              activation = None,
#                              kernel_initializer=tf.random_normal_initializer(),
#                              use_bias=False, name = "conv2")
#     conv2 = bn_wrapper(conv2, mode)

#     add_summery("maxpool: [3,3] kernal, strides: 2")
#     conv2 = tf.layers.max_pooling2d(conv2,3,2,"same")

#     conv2 = res_blocks(conv2, b_2, level, filters, mode,
#                        block_size = block_size, kernal_size=kernal_size, factorize = b_2_fac)


#     level = 3
#     # i: [-1, 32,32, fliterSize1]
#     # o: [-1, 16, 16, filtersize2]
#     add_summery("conv-{0}-0-0: size:{1}, kernal:{2}, filters: {3}, strides: 1".
#                 format(level,
#                        256/2**(level),
#                        3,
#                        filters * (2**level)))    
#     conv3 = tf.layers.conv2d(conv2,
#                              filters= filters * (2**level),
#                              kernel_size = 3,
#                              padding = "SAME",
# 			     strides = 1,
#                              activation = None,
#                              kernel_initializer=tf.random_normal_initializer(),
#                              use_bias=False, name = "conv3")
#     conv3 = bn_wrapper(conv3, mode)

#     add_summery("maxpool: [3,3] kernal, strides: 2")
#     conv3 = tf.layers.max_pooling2d(conv3,3,2,"same")
    

#     conv3 = res_blocks(conv3, b_3, level, filters, mode,
#                        block_size = block_size, kernal_size=kernal_size, factorize=b_3_fac)
    

#     level = 4
#     # i: [-1, 16,16, fliterSize1]
#     # o: [-1, 8, 8, filtersize2]
#     add_summery("conv-{0}-0-0: size:{1}, kernal:{2}, filters: {3}, strides: 1".
#                 format(level,
#                        256/2**(level),
#                        3,
#                        filters * (2**level)))    
#     conv4 = tf.layers.conv2d(conv3,
#                              filters= filters * (2**level),
#                              kernel_size = 3,
#                              padding = "SAME",
# 			     strides = 1,
#                              activation = None,
#                              kernel_initializer=tf.random_normal_initializer(),
#                              use_bias=False, name = "conv4")
#     conv4 = bn_wrapper(conv4, mode)

#     add_summery("maxpool: [3,3] kernal, strides: 2")
#     conv4 = tf.layers.max_pooling2d(conv4,3,2,"same")

#     conv4 = res_blocks(conv4, b_4, level, filters, mode,
#                        block_size = block_size, kernal_size=kernal_size, factorize = b_4_fac)

#     add_summery("conv: size:1, stride:1, padding:same, filters:2048")
#     # i: [-1, 8, 8, 512]
#     # o: [-1, 8, 8, 1024]
#     conv5 = tf.layers.conv2d(conv4,
#                              filters= 2048,
#                              kernel_size = 1,
#                              padding = "same",
#         		     strides = 1,
#                              activation = None,
#                              kernel_initializer=tf.random_normal_initializer(),
#                              use_bias=False, name = "conv5-dimred")
#     conv5 = bn_wrapper(conv5, mode)
    
#     # level = 5
#     #  # i: [-1, 8, 8, 128]
#     #  # o: [-1, 6, 6, 1024]
#     # add_summery("conv: size:3, stride:1, padding:valid, filters: 1024")
#     # conv5 = tf.layers.conv2d(conv5,
#     #                          filters= filters * (2**level),
#     #                          kernel_size = 3,
#     #                          padding = "valid",
#     #     		     strides = 1,
#     #                          activation = None,
#     #                          kernel_initializer=tf.random_normal_initializer(),
#     #                          use_bias=False, name = "conv5-0")
#     # conv5 = bn_wrapper(conv5, mode)


#     # level=6
#     # add_summery("conv: size:3, stride:1, padding:valid")
#     # # i: [-1, 6, 6, 1024]
#     # # o: [-1, 4, 4, 2048]
#     # conv6 = tf.layers.conv2d(conv5,
#     #                          filters= filters * (2**level),
#     #                          kernel_size = 3,
#     #                          padding = "valid",
#     #     		     strides = 1,
#     #                          activation = None,
#     #                          kernel_initializer=tf.random_normal_initializer(),
#     #                          use_bias=False, name = "conv6-0")
#     # conv6 = bn_wrapper(conv6, mode)
#     #add_summery("conv: size:3, stride:1, padding:valid")
#     #conv5 = tf.layers.conv2d(conv4, 2048, 3, 1, padding = "valid", name="conv5-1")
    
#     add_summery("average pool: size: 8, stride: 1, padding:valid")
#     pool = tf.layers.average_pooling2d(conv5,
#                                        pool_size = 8,
#                                        strides=1,
#                                        padding='valid',
#                                        name="pool")
                                       
#     # Flatten tensor into a batch of vectors
#     # Input Tensor Shape: [batch_size, 7, 7, 64]
#     # Output Tensor Shape: [batch_size, 7 * 7 * 64]
#     flat = tf.reshape(pool, [-1, 2048], name = "flat")
    
#     # Dense Layer
#     # Densely connected layer with 1024 neurons
#     # Input Tensor Shape: [batch_size, 7 * 7 * 64]
#     # Output Tensor Shape: [batch_size, 1024]
#     #add_summery("Dense1: [2000]")

#     add_summery("Dense: 1000")
#     dense = tf.layers.dense(flat, units=1000, activation=None, use_bias = False,
#                             name = "dense")
#     dense = bn_wrapper(dense, mode)
   
#     # add_summery("Logits")
#     # Logits layer
#     # Input Tensor Shape: [batch_size, 1024]
#     # Output Tensor Shape: [batch_size, 10]
#     add_summery("Logits: {0}".format(depth))
#     logits = tf.layers.dense(inputs=dense, units=depth, name = "logits-{0}".format(depth))
#     model_summery_finalize()
#     print(conv0.get_shape())
#     print(conv1.get_shape())
#     print(conv2.get_shape())
    
#     print(conv4.get_shape())
#     print(pool.get_shape())
#     print(flat.get_shape())
#     print(dense.get_shape())
#     #print(s.run(labels))
#     print(logits.get_shape())
#     print(MODEL_SUMMERY)
#     #print(s.run(tf.tens

#     update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

#     loss = None
#     train_op = None
#     tot = tf.Variable(0)
#     count = tf.Variable(0)
#     # Calculate Loss (for both TRAIN and EVAL modes)
#     if mode != tf.estimator.ModeKeys.PREDICT:
#       #label_set = [x for x in range(101)]
#       indices = tf.cast(labels, tf.int32)
#       #indices = tf.Print(indices, [indices], "The ONE HOT")
#       onehot_labels = tf.one_hot(indices= indices, depth=depth, axis = -1)
      
#       #print(onehot_labels.get_shape())
#       #print(labels.get_shape())
#       #loss = tf.losses.sparse_softmax_cross_entropy(labels = labels, logits = logits)
#       #logits = tf.Print(logits, [logits], "logits::")
#       loss = tf.losses.softmax_cross_entropy(
#         onehot_labels=tf.reshape(onehot_labels, [-1, depth]), logits=logits)
#       #loss = tf.Print(loss, [loss], "THE LOSS::::")
      
#       # Cnfigure the Training Op (for TRAIN mode)
#     if mode == tf.estimator.ModeKeys.TRAIN:
#       with tf.control_dependencies(update_ops):
#         train_op = tf.train.AdamOptimizer(learning_rate =
#                                           _LEARNING_RATE).minimize(loss,
#                                                                    tf.train.get_global_step())
        
#     # train_op = tf.contrib.layers.optimize_loss(
#     #     loss=loss,
#     #     global_step=tf.contrib.framework.get_global_step(),
#     #     learning_rate=0.005,
#     #     optimizer="SGD")
#     # Generate Predictions

    
    
#     predictions = {
#       "classes": tf.argmax(
#         input=logits, axis=1),
#       "probabilities": tf.nn.softmax(
#         logits, name="softmax_tensor")
#     }

#     export_outputs = {
#       "probabilities" : tf.estimator.export.ClassificationOutput(predictions["probabilities"])
#     }  
#     # print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
#     # print(predictions["probabilities"].get_shape())
#     if mode != tf.estimator.ModeKeys.PREDICT:
#       #labels = tf.Print(labels, [labels, predictions["probabilities"]], "YAYAY:    ", first_n=10)
#       #print("YAYAYAYAY:", labels.get_shape())
#       tkp=tf.nn.in_top_k(predictions["probabilities"], labels, k, name="tkp")
#       tot = tot + tf.count_nonzero(tkp, dtype=tf.int32)
#       count = count + tf.shape(tkp)[0]
#       tf.Print(logits, [tkp, tot, count]) 
#       metric = {"accuracy": tf.metrics.accuracy(labels, predictions["classes"]),
#                 "top_{0}_accuracy".format(k): (tot/count,  tf.no_op())}
#     else:
#       metric={}
#     # Return a ModelFnOps object
#     return tf.estimator.EstimatorSpec(
#       mode=mode, predictions=predictions, loss=loss, train_op=train_op, eval_metric_ops = metric,
#       export_outputs=export_outputs)
#   return model_fn
