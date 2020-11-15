from __future__ import print_function
from PIL import Image
import cv2
import os
import numpy as np
import pickle as cPickle
import gzip
import torch.utils.data as data
#import urllib
import urllib.request
import scipy.io

class MNISTEDGE(data.Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None):
        self.url = 'http://deeplearning.net/data/mnist/mnist.pkl.gz'
        self.filename = 'mnist.pkl.gz'
        self.filename_train_domain_1 = "mnist_train_original.pickle"
        self.filename_train_domain_2 = "mnist_train_edge.pickle"
        self.filename_test_domain_1 = "mnist_test_original.pickle"
        self.filename_test_domain_2 = "mnist_test_edge.pickle"
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.download()
        self.create_two_domains()
        # now load the picked numpy arrays
        #if self.train:
        filename_train_domain_1 = os.path.join(self.root, self.filename_train_domain_1)
        filename_train_domain_2 = os.path.join(self.root, self.filename_train_domain_2)
        filename_test_domain_1 = os.path.join(self.root, self.filename_test_domain_1)
        filename_test_domain_2 = os.path.join(self.root, self.filename_test_domain_2)
        data_a, labels_a = cPickle.load(gzip.open(filename_train_domain_1, 'rb'))
        data_b, labels_b = cPickle.load(gzip.open(filename_train_domain_2, 'rb'))
        testdata_a, testlabels_a = cPickle.load(gzip.open(filename_test_domain_1, 'rb'))
        testdata_b, testlabels_b = cPickle.load(gzip.open(filename_test_domain_2, 'rb'))

        #####################
#        pic_data1 = [data[0] for data in testdata_a]
#        scipy.io.savemat('data_a.mat', mdict={'pic_data1':pic_data1})
#        pic_data2 = [data[0] for data in testdata_b]
#        scipy.io.savemat('data_b.mat', mdict={'pic_data2':pic_data2})
#        scipy.io.savemat('label_a.mat', mdict={'label_a':testlabels_a})
#        scipy.io.savemat('label_b.mat', mdict={'label_b':testlabels_b})
#        print('saved...')
        self.paried_num = 2000-(round((2000-2000*0.9)/3))*3
        data_0 = scipy.io.loadmat('rand1/paired_a3_90.mat')
        data_dict=dict(data_0)
        data_1 = data_dict['xpaired']
        data_2 = scipy.io.loadmat('rand1/paired_b3_90.mat')
        data_dict=dict(data_2)
        data_3 = data_dict['ypaired']
        data_4 = scipy.io.loadmat('rand1/paired_c3_90.mat')
        data_dict=dict(data_4)
        data_5 = data_dict['zpaired']
        data_a = np.zeros((self.paried_num, 1, 216))
        data_b = np.zeros((self.paried_num, 1, 76))
        data_c = np.zeros((self.paried_num, 1, 64))
        for i in range(self.paried_num):
            data_a[i][0]=data_1[i]
        for i in range(self.paried_num):
            data_b[i][0]=data_3[i]
        for i in range(self.paried_num):
            data_c[i][0]=data_5[i]
#        testdata_a = data_a
#        testdata_b = data_b
        #########################
#        self.train_data_a = data_a# * 255.0
#        self.train_labels_a = labels_a
#        self.train_data_b = data_b# * 255.0
#        self.train_labels_b = labels_b
#        self.test_data_a = testdata_a# * 255.0
#        self.test_labels_a = testlabels_a
#        self.test_data_b = testdata_b #* 255.0
#        self.test_labels_b = testlabels_b
#        self.train_data_a = self.train_data_a.transpose((0, 2, 3, 1))  # convert to HWC
#        self.train_data_b = self.train_data_b.transpose((0, 2, 3, 1))  # convert to HWC
#        #self.train_data_a = self.train_data_a.reshape(paried_num,1,784)
#        #self.train_data_b = self.train_data_b.reshape(paried_num,1,784)
#        self.test_data_a = self.test_data_a.transpose((0, 2, 3, 1))  # convert to HWC
#        self.test_data_b = self.test_data_b.transpose((0, 2, 3, 1))  # convert to HWC
        self.train_data_a = data_a# * 255.0
        self.train_labels_a = labels_a
        self.train_data_b = data_b# * 255.0
        self.train_labels_b = labels_b
        self.train_data_c = data_c# * 255.0
        self.train_labels_b = labels_b
#        self.test_data_a = data_a# * 255.0
#        self.test_labels_a = testlabels_a
#        self.test_data_b = data_b #* 255.0
#        self.test_labels_b = testlabels_b       
#        self.test_data_c = data_c #* 255.0
#        self.test_labels_b = testlabels_b

        print(self.train_data_a.shape)
        print(self.train_data_b.shape)
        print(self.train_data_c.shape)
#        print(self.test_data_a.shape)
#        print(self.test_data_b.shape)

        
    def __getitem__(self, index):
        #index_2 = np.random.randint(0, self.__len__(), 1)
        if self.train:
            img_a, img_b, img_c = self.train_data_a[index, :], self.train_data_b[index, :], self.train_data_c[index, :]
        #elif self.test:
        #    img_a, img_b = self.test_data_a[index, ::], self.test_data_b[index_2, ::].squeeze(axis=0)
        else:
            img_a, img_b, img_c = self.test_data_a[index, :], self.test_data_b[index, :], self.train_data_c[index, :]
        #elif self.test:
            ############################################################
            #lab_a, lab_b = self.test_labels_a[index], self.test_labels_b[index]
            ############################################################

        if self.transform is not None:
            img_a = self.transform(img_a)
            img_b = self.transform(img_b)
            img_c = self.transform(img_c)

        ############################################################
        return img_a, img_b, img_c#, lab_a, lab_b
        ############################################################
    def __len__(self):
        if self.train:
            return self.paried_num
        else:
            return self.paried_num

    def download(self):
        filename = os.path.join(self.root, self.filename)
        if os.path.isfile(filename):
            return
        dirname = os.path.dirname(filename)
        if not os.path.isdir(dirname):
            os.mkdir(dirname)
        print("Download %s to %s" % (self.url, filename))
        urllib.request.urlretrieve(self.url, filename)
        print("[DONE]")
        return

    def create_two_domains(self):

        def save_domains(input_data, input_labels, domain_1_filename, domain_2_filename, domain_1_filename_test, domain_2_filename_test):
            n_samples = input_data.shape[0]
            test_samples = int(n_samples/10)
            arr = np.arange(n_samples)
            np.random.shuffle(arr)
            data_a = np.zeros((n_samples - test_samples, 1, 28, 28))
            label_a = np.zeros(n_samples - test_samples, dtype=np.int32)
            data_b = np.zeros((n_samples - test_samples, 1, 28, 28))
            label_b = np.zeros(n_samples - test_samples, dtype=np.int32)
            test_data_a = np.zeros((test_samples, 1, 28, 28))
            test_label_a = np.zeros(test_samples, dtype=np.int32)
            test_data_b = np.zeros((test_samples, 1, 28, 28))
            test_label_b = np.zeros(test_samples, dtype=np.int32)


            for i in range(0, n_samples - test_samples):
                img = input_data[arr[i], :].reshape(28, 28)
                label = input_labels[arr[i]]
                dilation = cv2.dilate(img, np.ones((3, 3), np.uint8), iterations=1)
                edge = dilation - img
                data_a[i, 0, :, :] = img
                data_b[i, 0, :, :] = edge
                label_a[i] = label
                label_b[i] = label

            for i in range(n_samples - test_samples, n_samples):
                img = input_data[arr[i], :].reshape(28, 28)
                label = input_labels[arr[i]]
                dilation = cv2.dilate(img, np.ones((3, 3), np.uint8), iterations=1)
                edge = dilation - img
                test_data_a[i - (n_samples - test_samples), 0, :, :] = img
                test_data_b[i - (n_samples - test_samples), 0, :, :] = edge
                test_label_a[i - (n_samples - test_samples)] = label
                test_label_b[i - (n_samples - test_samples)] = label



            with gzip.open(domain_1_filename, 'wb') as handle:
                cPickle.dump((data_a, label_a), handle)
            with gzip.open(domain_2_filename, 'wb') as handle:
                cPickle.dump((data_b, label_b), handle)
            with gzip.open(domain_1_filename_test, 'wb') as handle:
                cPickle.dump((test_data_a, test_label_a), handle)
            with gzip.open(domain_2_filename_test, 'wb') as handle:
                cPickle.dump((test_data_b, test_label_b), handle)


        filename = os.path.join(self.root, self.filename)
        filename_train_domain_1 = os.path.join(self.root, self.filename_train_domain_1)
        filename_train_domain_2 = os.path.join(self.root, self.filename_train_domain_2)
        filename_test_domain_1 = os.path.join(self.root, self.filename_test_domain_1)
        filename_test_domain_2 = os.path.join(self.root, self.filename_test_domain_2)
        if os.path.isfile(filename_train_domain_1) and os.path.isfile(filename_train_domain_2) \
                and os.path.isfile(filename_test_domain_1) and os.path.isfile(filename_test_domain_2):
            return
        f = gzip.open(filename, 'rb')
        train_set, valid_set, test_set = cPickle.load(f, encoding='latin1')
        f.close()
        #images = train_set[0]
        #labels = train_set[1]

        images = np.concatenate((train_set[0], valid_set[0]), axis=0)
        labels = np.concatenate((train_set[1], valid_set[1]), axis=0)
        print("Compute edge images")
        print("Save origin to %s and edge to %s" % (filename_train_domain_1, filename_train_domain_2))
        save_domains(images, labels, filename_train_domain_1, filename_train_domain_2, filename_test_domain_1, filename_test_domain_2)
        print("[DONE]")


