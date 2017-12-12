import numpy as np
import tensorflow as tf
import os
import random
import json
import csv
import cv2
CLASSES_JSON = "../Datasets/meta/classes-msi.json"
TRAIN_JSON = "../Datasets/meta/train-msi.json"
TEST_JSON = "../Datasets/meta/test-msi.json"


def generate_combined_json(json_files,CLASSES_JSON=None, TRAIN_JSON=None, TEST_JSON=None):
    ''' generate_combined_json([["../Datasets/Food-101/food-101/meta/classes-msi.json","../Datasets/Food-101/food-101/meta/train10-msi.json","../Datasets/Food-101/food-101/meta/test10-msi.json"], ["../Datasets/meta/instagramdownload/classes-msi.json", "../Datasets/meta/instagramdownload/train-msi.json","../Datasets/meta/instagramdownload/test-msi.json"]])
'''
    classes_json_combined = {}
    classes_combined=[]
    classes_index = 0
    train_json_combined = []
    test_json_combined = []
    for classes_json, train_json_file, test_json_file  in json_files:
        classes_dict = json.load(open(classes_json), object_hook=jsonKeys2str)
        train_json = json.load(open(train_json_file), object_hook = jsonKeys2str)
        test_json = json.load(open(test_json_file), object_hook = jsonKeys2str)
        for k,v in classes_dict.items():
            if v not in classes_combined:
                classes_json_combined[classes_index] = v
                classes_combined.append(v)
                classes_index +=1

        classes_json_combined_switched = {v:k for k,v in classes_json_combined.items()}
        print(train_json[:10])
        for entry in train_json:
            train_json_combined.append([entry[0],
                                        classes_json_combined_switched[classes_dict[entry[1]]]])
        for entry in train_json:
            train_json_combined.append([entry[0],
                                       classes_json_combined_switched[classes_dict[entry[1]]]])
    json.dump(classes_json_combined, open(CLASSES_JSON, "w"))
    json.dump(train_json_combined, open(TRAIN_JSON, "w"))
    json.dump(test_json_combined, open(TEST_JSON, "w"))
                

def generate_json_gen(csv_file, files_prefix_path, CLASSES_JSON=None, TRAIN_JSON=None, TEST_JSON=None,labels_list=["rice_and_curry"], labels_list_ignore=False, only_print_labels=False):
    all_files= {}
    #all_uncoded_labels= {}
    labels = set([])
    csv_file = csv.reader(open(csv_file))
    i = 0
    for row in csv_file:
        label = row[1].split("|")[0]
        try:
            all_files[label].append(os.path.join(files_prefix_path, row[0]))
        except KeyError:
            all_files[label] = [os.path.join(files_prefix_path, row[0])]
            #all_uncoded_labels.append(label)
        labels.add(label)
    if only_print_labels:
        print(labels)
        return
    
    i = 0
    label_filtered = []
    for label in labels:
        if labels not in labels_list and labels_list_ignore:
            label_filtered.append(label)
        elif label in labels_list and not labels_list_ignore:
            label_filtered.append(label)
    data_codes = get_data_codes(label_filtered, CLASSES_JSON)
    print(data_codes)
    json.dump(data_codes, open(CLASSES_JSON, "w"))
    try:
        # train_json_list = [entry for entry in json.load(open(TRAIN_JSON))
        #                    if entry[0] not in all_files[entry[1]]]
        train_json_list =[]
        for entry in json.load(open(TRAIN_JSON)):
            #print(entry)
            try:
                if entry[0] not in all_files[data_codes[entry[1]]]:
                    train_json_list.append(entry)
            except:
                train_json_list.append(entry)
    except FileNotFoundError:
        train_json_list = []
    try:
        test_json_list =  []
        for entry in json.load(open(TEST_JSON)):
            #print(entry)
            try:
                if entry[0] not in all_files[data_codes[entry[1]]]:
                    test_json_list.append(entry)
            except:
                test_json_list.append(entry)
    except FileNotFoundError:
        test_json_list = []

    #print(len(train_json_list), len(test_json_list))
    for k,v in all_files.items():
        if k in label_filtered:
            
            labeled_set = gen_labled_set([k]*len(v), v, data_codes)
            print(len(labeled_set))
            length = int(len(labeled_set)*0.1)
            train_json_list += labeled_set[length:]
            test_json_list += labeled_set[:length]
    json.dump(train_json_list, open(TRAIN_JSON, "w"))
    json.dump(test_json_list, open(TEST_JSON, "w"))
    #print(len(train_json_list), len(test_json_list))
    
    #gen_labled_set(all_uncoded_labels, all_files, data_codes),
    
    # for i in x[:20]:
    #     cv2.imshow("a", cv2.imread(i[0]))
    #     cv2.waitKey(300)
    # cv2.destroyAllWindows()
        
def generate_json_101(DATA_FILE_LOCATION = "/media/Files/Research/FoodClassification/Datasets/Food-101/food-101/images/",
                      TRAIN_JSON = TRAIN_JSON,
                      TEST_JSON = TEST_JSON,
                      CLASSES_JSON = CLASSES_JSON):
    
    train_files = []
    test_files = []
    train_uncoded_labels=[]
    test_uncoded_labels=[]
    test_size = 1000 * 0.1
    labels_set=set([])
    for rdir, sd, fileList in os.walk(DATA_FILE_LOCATION):
        print("New set")
        i = 0
        for f in fileList:
            # if i > 50:
            #   break
            if f.lower().endswith((".jpg",".jpeg")):
                if i > test_size:
                    train_files.append(os.path.abspath(os.path.join(rdir,f)))
                    train_uncoded_labels.append(rdir.split("/")[-1])
                else:
                    test_files.append(os.path.abspath(os.path.join(rdir,f)))
                    test_uncoded_labels.append(rdir.split("/")[-1])

                if i%50 == 0:
                    print("{0}--{1}".format(rdir.split("/")[-1],rdir.split("/")[-1]))
                    print("\t{0}::{1}".format(rdir, f))
                labels_set.add(rdir.split("/")[-1])

                
                i += 1
            
    data_codes = {}
    i = 0
    for label in labels_set:
        data_codes[i]=label
        i += 1
    json.dump(data_codes, open(CLASSES_JSON, "w"))

    
    
    json.dump(gen_labled_set(train_uncoded_labels, train_files),
              open(TRAIN_JSON, "w"))
    json.dump(gen_labled_set(test_uncoded_labels, test_files),
              open(TEST_JSON, "w"))
    print(data_codes[53])

def gen_labled_set(labels, files, data_codes):
    return_list = []
    for i in range(len(labels)):
        code = -1
        for k,v in data_codes.items():
            if v == labels[i]:
                code = k
                break
        if code != -1:
            return_list.append([files[i],code])
    return return_list #list(zip(files, labels))

def get_data_codes(labels_set, CLASSES_JSON):
    try:
        data_codes = json.load(open(CLASSES_JSON), object_hook = jsonKeys2str)
        print("loading pre saved data codes and appending")
    except FileNotFoundError:
        data_codes = {}
        print("creating new data codes")
    finally:
        coded_labels = [v for k,v in data_codes.items()]
        uncoded_labels = [l for l in labels_set if l not in coded_labels]
        i = len(data_codes)
        for label in uncoded_labels:
            data_codes[i]=label
            i+=1
    return data_codes
        
        
def jsonKeys2str(x):
    if isinstance(x, dict):
        return {int(k):v for k,v in x.items()}
    return x

class DataLoader():
    summery = "Load images on demand and decide classes when loaded"
    train_files = []
    train_labels = []
    test_files = []
    test_labels = []
    DATA_CODE_MAPPING = {}
    def __init__(self, train_file_location,
                 test_file_location = None,
                 batch_size = 3,
                 use_all_classes = True,
                 classes_count = None):
        if not use_all_classes and classes_count is None:
            raise TypeError("If 'use_all_classes' is False, 'classes_count' must be given")
        self.classes_count = classes_count
        self.train_file_location = train_file_location
        self.test_file_location = test_file_location
        self.train_files, self.train_labels, self.test_files, self.test_labels= self._load_data(train_file_location, test_file_location)
        self.batch_size = batch_size
    
    def _load_files(self, file_location, use_all_classes):
        files=[]
        uncoded_labels=[]
        labels_set=set([])
        j = 0
        for rdir, sd, fileList in os.walk(file_location):
            if not use_all_classes and j > self.classes_count:
                break
            j += 1
            for f in fileList:
                
                # if i > 50:
                #   break
                if f.lower().endswith((".jpg",".jpeg")):
                    files.append(os.path.abspath(os.path.join(rdir,f)))
                    
                    uncoded_labels.append(rdir.split("/")[-1])
                    labels_set.add(rdir.split("/")[-1])
        return files,uncoded_labels, labels_set
                    
      
    def _load_data(self, train_file_location, test_file_location=None, use_all_classes=True):
        labels_set = set()
        train_files = []
        test_files = []
        train_uncoded_labels = []
        test_uncoded_labels = []
        train_labels = []
        test_labels = []

        train_files, train_uncoded_labels, labels_set = self._load_files(train_file_location,
                                                                         use_all_classes)

        
        i = 0
        train_labels = [0 for x in range(len(train_uncoded_labels))]
        for label in labels_set:
            self.DATA_CODE_MAPPING[i] = label
            i = i + 1
            
        for i in range(len(train_uncoded_labels)):
            for code, label in self.DATA_CODE_MAPPING.items():
                if train_uncoded_labels[i] == label:
                    train_labels[i] = code
                    #print(label)
                    #print(code)
                    break

        test_labels = [0 for x in range(len(test_uncoded_labels))]
        if test_file_location!=None:
            test_files, test_uncoded_labels, test_labels_set = self._load_files(train_file_location,
                                                                                use_all_classes)
            #need to handel if test set contains labels not in train

            for i in range(len(test_uncoded_labels)):
                for code, label in self.DATA_CODE_MAPPING.items():
                    if test_uncoded_labels[i] == label:
                        test_labels[i] = code
                        #print(label)
                        #print(code)
                        break
                    
                    #print(type(len(all_feiles)))
        else:
            all_files = train_files
            all_labels = train_labels
            partitions = [1]*len(all_files)
            test_size = int(float(len(all_files))*0.1)
            partitions[:test_size] = [0]*test_size
            random.shuffle(partitions)
  
            #print(all_uncoded_labels)
            
            
            test_files = [file for file,index in zip(all_files, partitions) if not index]
            train_files = [file for file,index in zip(all_files, partitions) if index]
            test_labels = [label for label,index in zip(all_labels, partitions) if not index]
            train_labels = [label  for label,index in zip(all_labels, partitions) if index]

        self.classes_count = len(labels_set)
        print("Total files: {0}".format(len(train_files) + len(test_files)))
        print("Train files: {0}".format(len(train_files)))
        print("Test files: {0}".format(len(test_files)))
        print("Classes: {0}".format(self.classes_count))
        return train_files, train_labels, test_files, test_labels

    def set_classes(self, use_all_classes, classes_count):
        if not use_all_classes and classes_count is None:
            raise TypeError("If 'use_all_classes' is False, 'classes_count' must be given")
        self.classes_count = classes_count
        self._load_data(self.train_file_location, self.test_file_location, use_all_classes)
        
    def get_train_input_fn(self, mode= tf.estimator.ModeKeys.TRAIN):
        
        #test_files, train_files = tf.dynamic_partition(all_files, partitions, 2)
        #test_labels, train_labels = tf.dynamic_partition(all_labels, partitions, 2)
        if self.train_labels is None or self.train_files is None:
            self.set_classes(self.use_all_classes, self.classes_count)
        def input_fn():
            train_input_queue = tf.train.slice_input_producer([self.train_files, self.train_labels],
                                                              shuffle = True)
            
            #file_name = tf.Print(train_input_queue[0], [train_input_queue[0]], "Tha file name::")
            file_content = tf.read_file(train_input_queue[0])
            train_data = tf.image.resize_images(tf.image.decode_jpeg(file_content, channels = 3),
                                          [512,512])
            train_label = train_input_queue[1]
            
            train_data_batch, train_label_batch = tf.train.shuffle_batch([train_data, train_label],
                                                                         batch_size = self.batch_size,
                                                                         capacity = 200,
                                                                         min_after_dequeue = 150)
            if mode is tf.estimator.ModeKeys.TRAIN:
                return train_data_batch, train_label_batch
            else:
                return train_data, train_label
        return input_fn

    def get_test_input_fn(self):
        if self.test_labels is None or self.test_files is None:
            self.set_classes(self.use_all_classes, self.classes_count)
        def input_fn():
            test_input_queue = tf.train.slice_input_producer([self.test_files, self.test_labels],
                                                             shuffle = True)
            #file_name = tf.Print(train_input_queue[0], [train_input_queue[0]], "Tha file name::")
            file_content = tf.read_file(test_input_queue[0])
            test_data = tf.image.resize_images(tf.image.decode_jpeg(file_content, channels = 3),
                                               [512,512])
            test_label = test_input_queue[1]
            return test_data, test_label
        return input_fn


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
        
    def get_train_input_fn(self, mode= tf.estimator.ModeKeys.TRAIN):
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
            if mode == tf.estimator.ModeKeys.TRAIN:
                batch_size = self.batch_size
            else:
                batch_size = 1
            train_data_batch, train_label_batch = tf.train.shuffle_batch([train_data, train_label],
                                                                         batch_size = batch_size,
                                                                         capacity = 200,
                                                                         min_after_dequeue = 150)
            #train_label_batch = tf.Print(train_label_batch, [train_label_batch], "Yalaaaa:")
            return train_data_batch, train_label_batch
            # if mode == tf.estimator.ModeKeys.TRAIN:
            #     return train_data_batch, train_label_batch
            # else:
            #     return train_data, train_label
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


class cifar10Loader():
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
    
    def get_train_input_fn(self, mode = tf.estimator.ModeKeys.TRAIN):
        def input_fn():
            train_data_tensor = tf.convert_to_tensor(self.train_data)
            train_input_queue = tf.train.slice_input_producer(
                [train_data_tensor, self.train_labels], shuffle=False)
            train_image_resized = tf.image.resize_images(train_input_queue[0],
                                                         [256,256])
            train_data_batch, train_labels_batch = tf.train.batch([train_image_resized, train_input_queue[1]],
                                                                  batch_size = self.batch_size)
            # if mode == tf.estimator.ModeKeys.TRAIN:
            return train_data_batch, train_labels_batch
            # else:
            #     return train_image_resized, train_input_queue[1]
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
