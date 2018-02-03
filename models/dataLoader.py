import numpy as np
import tensorflow as tf
import os
import random
import json
import csv
import cv2
from helper import DataLoader
from utils import ModeKeys


CLASSES_JSON = "../Datasets/meta/classes-msi.json"
TRAIN_JSON = "../Datasets/meta/train-msi.json"
TEST_JSON = "../Datasets/meta/test-msi.json"


def jsonKeys2str(x):
    if isinstance(x, dict):
        return {int(k):v for k,v in x.items()}
    return x

class DataLoaderJson(DataLoader):
    summery= "classes are as defined in a json file to ensure same classes can be loaded with same labels during each run"
    train_files = None
    train_labels = None
    test_files = None
    test_labels = None
    DATA_CODE_MAPPING = {}
    def __init__(self,
                 classes_file_location,
                 train_file_location,
                 test_file_location,
                 batch_size = 3,
                 use_all_classes = True,
                 classes_count = None,
                 classes_offset = 0):
        if not use_all_classes and classes_count is None:
            raise TypeError("If 'use_all_classes' is False, 'classes_count' must be given")
        self.DATA_CODE_MAPPING = json.load(open(classes_file_location),
                                           object_hook=self._jsonKeys2str)
        self.default_classes_offset = classes_offset
        self.default_use_all_classes=use_all_classes
        self.default_classes_count=classes_count
        self.use_all_classes = use_all_classes
        self.classes_count = classes_count
        self.train_data,  self.test_data= self._load_data(train_file_location, test_file_location)
        self.batch_size = batch_size
        self.summery= "classes are as defined in a json file to ensure same classes can be loaded with same labels during each run"
                  
      
    def _load_data(self, train_file_location, test_file_location):
        train_files = []
        test_files = []
        train_labels = []
        test_labels = []


        train_data = json.load(open(train_file_location))
        test_data = json.load(open(test_file_location))
        
        return train_data, test_data


    def _jsonKeys2str(self, x):
        if isinstance(x, dict):
            return {int(k):v for k,v in x.items()}
        return x

    def set_classes(self,
                    use_all_classes = None,
                    classes_count = None,
                    classes_offset = None):
        if use_all_classes is None:
            print("using default value for 'use_all_classes'")
            use_all_classes = self.default_use_all_classes
        if classes_count is None:
            print("using default value for 'classes_count'")
            classes_count = self.default_classes_count
        if classes_offset is None:
            print("using default value for 'classes_offset'")
            classes_offset = self.default_classes_offset
        OFFSET = classes_offset
            
        if classes_count is not None and use_all_classes:
            print("Ignoring 'classes_count'")
            
        if not use_all_classes and classes_count is None:
            raise TypeError("If 'use_all_classes' is False, 'classes_count' must be given")
        if use_all_classes:
            classes_count=len(self.DATA_CODE_MAPPING)

        self.use_all_classes=use_all_classes
        if self.use_all_classes or isinstance(OFFSET, int):
            if not isinstance(OFFSET, int):
                OFFSET = 0
            self.classes_count=classes_count
            self.train_files = [f for f,l in self.train_data
                                if l < classes_count+OFFSET and l > OFFSET-1]
            self.train_labels = [l-OFFSET for f,l in self.train_data
                                 if l < classes_count+OFFSET and l > OFFSET-1]
            self.test_files = [f for f,l in self.test_data
                               if l < classes_count+OFFSET and l > OFFSET-1]
            self.test_labels = [l-OFFSET for f,l in self.test_data
                                if l < classes_count+OFFSET and l > OFFSET-1]
            print(set(self.test_labels))
            print(set(self.train_labels))
            self.used_labels = set([l for f,l in self.test_data
                                    if l < classes_count+OFFSET and l > OFFSET-1])
        else:
            
            self.used_labels =  self._get_labels_from_names(OFFSET)
            self.classes_count = len(self.used_labels)
            self.train_files = [f for f,l in self.train_data
                                if l in self.used_labels]
            self.train_labels = [l for f,l in self.train_data
                                 if l in self.used_labels]
            self.test_files = [f for f, l in self.test_data
                               if l in self.used_labels]
            self.test_labels = [l for f, l in self.test_data
                               if l in self.used_labels]
            
            
        print("Labels used: {0}".format(self.used_labels))
        print("Total Files: {0}".format(len(self.train_files) + len(self.test_files)))
        print("Train Files: {0}".format(len(self.train_files)))
        print("Test Files: {0}".format(len(self.test_files)))

    def _get_labels_from_names(self, names_list):
        label_list = [(k,v) for k,v in self.DATA_CODE_MAPPING.items() if v in names_list]
        if len(label_list) < len(names_list):
            missing_names = [n for n in names_list if n not in [v for k,v in label_list]]
            raise Exception("Missing labels: {0}".format(missing_names))
        else:
            return [k for k,v in label_list]
            
        


class DataLoaderJsonSized(DataLoaderJson):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)
        self.summery = self.summery + \
                       "\ncroped to samlled size and resized to 256\n \
                       randomly randomcrop or center crop, and random flips"
        
    def get_train_input_fn(self, mode= ModeKeys.TRAIN):
        _size = 470 
        #test_files, train_files = tf.dynamic_partition(all_files, partitions, 2)
        #test_labels, train_labels = tf.dynamic_partition(all_labels, partitions, 2)
        if self.train_labels is None or self.train_files is None:
            self.set_classes(self.use_all_classes, self.classes_count)
        def input_fn():
            train_input_queue = tf.train.slice_input_producer([self.train_files, self.train_labels],
                                                              shuffle = True)
            
            #file_name = tf.Print(train_input_queue[0], [train_input_queue[0]], "Tha file name::")
            file_content = tf.read_file(train_input_queue[0])
            train_image = tf.image.decode_jpeg(file_content, channels = 3)
            
            size = tf.shape(train_image)
            #size = tf.Print(size, [train_input_queue[1]], "Yala:")
            #size_ = tf.minimum(size[0],size[1])
            random_val1 = tf.random_uniform([1], maxval = 10)[0]
            random_val2 = tf.random_uniform([1], maxval = 10)[0]

            values=tf.convert_to_tensor([size[0], size[1], _size])
            i = tf.cast([tf.argmin(values)], tf.int32)[0]
            # ret_0 = lambda: tf.constant([0])
            # ret_1 = lambda: tf.constant([-1])
            # def ret_y(s):
            #     return lambda: tf.to_int32(tf.to_float(s - values[i]) * 0.5 *
            #                                tf.random_normal([1],
            #                                                 stddev=0.33333333333,
            #                                                 mean=1))
            # # y1 = tf.random_uniform([1], maxval= y1_maxval, dtype=tf.int32)
            # # x1 = tf.random_uniform([1], maxval= x1_maxval, dtyp)e=tf.int32)
            # y1 = tf.case([(tf.equal(0,i), ret_0)], default=ret_y(size[0]))
            # x1 = tf.case([(tf.equal(1,i), ret_0)], default=ret_y(size[1]))
            
            random_crop =lambda: tf.random_crop(train_image, [values[i], values[i], 3])
            center_crop =lambda: tf.image.resize_image_with_crop_or_pad(train_image,
                                                                        values[i],
                                                                        values[i])
            train_image_r1 = tf.case([(tf.less(random_val1, 9), random_crop)], center_crop)
            
            flip = lambda: tf.image.flip_left_right(train_image_r1)
            noflip = lambda: train_image_r1
            train_image_r2 = tf.case([(tf.less(random_val2, 5), flip)], noflip)
            # train_image_ = tf.Print(train_image, [tf.shape(train_image_r2)])
            # train_croped = tf.image.crop_and_resize(train_image,
            #                                        )
            train_data = tf.reshape(tf.image.resize_images(train_image_r2,
                                                          [256, 256]), [256, 256,3])
            train_label = train_input_queue[1]
            if mode == ModeKeys.TRAIN:
                batch_size = self.batch_size
            else:
                batch_size = 1
            train_data_batch, train_label_batch = tf.train.shuffle_batch([train_data, train_label],
                                                                         batch_size = batch_size,
                                                                         capacity = 200,
                                                                         min_after_dequeue = 150)
            #train_label_batch = tf.Print(train_label_batch, [train_label_batch], "Yalaaaa:")
            return train_data_batch, train_label_batch
        return input_fn

    def get_test_input_fn(self):
        if self.test_labels is None or self.test_files is None:
            self.set_classes(self.use_all_classes, self.classes_count)
        def input_fn():
            test_input_queue = tf.train.slice_input_producer([self.test_files, self.test_labels],
                                                             shuffle = True)
            #file_name = tf.Print(train_input_queue[0], [train_input_queue[0]], "Tha file name::")
            file_content = tf.read_file(test_input_queue[0])
            test_image = tf.image.decode_jpeg(file_content, channels = 3)
            # shape = tf.shape(test_image)
            # size = tf.minimum(shape[0], shape[1])
            # test_image = tf.image.resize_image_with_crop_or_pad(test_image,size, size)
            test_data = tf.image.resize_images(test_image,
                                               [256, 256])
            test_label = test_input_queue[1]
            test_data_batch, test_label_batch = tf.train.shuffle_batch([test_data, test_label],
                                                                         batch_size = 1,
                                                                         capacity = 200,
                                                                         min_after_dequeue = 150)
            #return test_data, test_label
            return test_data_batch, test_label_batch
        return input_fn


class cifar10Loader(DataLoader):
    def __init__(self,
                 classes_file_location,
                 train_file_location,
                 test_file_location,
                 batch_size = 3,
                 use_all_classes = True,
                 classes_count = None,
                 classes_offset = 0):
        self.batch_size = batch_size
        self.train_data, self.train_labels = self.processFiles(train_file_location)
        self.test_data, self.test_labels = self.processFiles(test_file_location)
        self.classes_count = 101
        self.summery = "cifar10 dataset, where each image is scaled to 256,256"
        print("CIFR10 FOR THE WIN!!!!!!!")
        self.test_files = self.test_data
        self.train_files = self.train_data
        print(len(self.test_files), len(self.train_files))
        self.used_labels = []
        self.DATA_CODE_MAPPING={}
        

    def processFiles(self, batch_files):
        file_batches = [self.unpickle(f) for f in batch_files]
        labels = np.concatenate([f[b"labels"] for f in file_batches])
        data = [img.reshape([3,32,32]).transpose(1,2,0)
                     for img in np.concatenate([f[b"data"] for f in file_batches])]
        
        return np.asarray(data), labels
        
    def unpickle(self, file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    def set_classes(self,
                    use_all_classes = None,
                    classes_count = None,
                    classes_offset = None):
        pass
    
    def get_train_input_fn(self, mode = ModeKeys.TRAIN):
        def input_fn():
            train_data_tensor = tf.convert_to_tensor(self.train_data)
            train_input_queue = tf.train.slice_input_producer(
                [train_data_tensor, self.train_labels], shuffle=False)
            train_image_resized = tf.image.resize_images(train_input_queue[0],
                                                         [256,256])
            train_data_batch, train_labels_batch = tf.train.batch([train_image_resized, train_input_queue[1]],
                                                                  batch_size = self.batch_size)
            return train_data_batch, train_labels_batch
        return input_fn

    def get_test_input_fn(self):
        def input_fn():
            test_data_tensor = tf.convert_to_tensor(self.test_data)
            test_input_queue = tf.train.slice_input_producer(
                [test_data_tensor, self.test_labels], shuffle=False)
            test_image_resized = tf.image.resize_images(test_input_queue[0],
                                                         [256,256])
            test_data_batch, test_labels_batch = tf.train.batch([test_image_resized, test_input_queue[1]],
                                                                  batch_size = self.batch_size)
            return test_data_batch, test_labels_batch# test_image_resized, test_input_queue[1]
        return input_fn
