#!/usr/bin/python
# encoding: utf-8

import random
import torch
from torch.utils.data import Dataset
from torch.utils.data import sampler
import torchvision.transforms as transforms
import lmdb
import six
import sys
import bisect
import warnings
from PIL import Image
import numpy as np
import string
import cv2
import os
import re

sys.path.append('../')
from utils import str_filt
from utils.labelmaps import get_vocabulary, labels2strs
from IPython import embed
from pyfasttext import FastText
random.seed(0)

from utils import utils_deblur
from utils import utils_sisr as sr
from utils import utils_image as util

scale = 0.90
kernel = utils_deblur.fspecial('gaussian', 15, 1.)
noise_level_img = 0.


def rand_crop(im):
    w, h = im.size
    p1 = (random.uniform(0, w*(1-scale)), random.uniform(0, h*(1-scale)))
    p2 = (p1[0] + scale*w, p1[1] + scale*h)
    return im.crop(p1 + p2)


def central_crop(im):
    w, h = im.size
    p1 = (((1-scale)*w/2), (1-scale)*h/2)
    p2 = ((1+scale)*w/2, (1+scale)*h/2)
    return im.crop(p1 + p2)


def buf2PIL(txn, key, type='RGB'):
    imgbuf = txn.get(key)
    buf = six.BytesIO()
    buf.write(imgbuf)
    buf.seek(0)
    im = Image.open(buf).convert(type)
    return im


class lmdbDataset(Dataset):
    def __init__(self, root=None, voc_type='upper', max_len=31, test=True):
        super(lmdbDataset, self).__init__()
        self.env = lmdb.open(
            root,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)

        if not self.env:
            print('cannot creat lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get(b'num-samples'))
            self.nSamples = nSamples

        self.max_len = max_len
        self.voc_type = voc_type

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index += 1
        txn = self.env.begin(write=False)

        label_key = b'label-%09d' % index
        word = str(txn.get(label_key).decode())

        try:
            img = buf2PIL(txn, b'image_hr-%09d' % index, 'RGB')
        except TypeError:
            img = buf2PIL(txn, b'image-%09d' % index, 'RGB')
        except IOError or len(label) > self.max_len:
            return self[index + 1]

        label_str = str_filt(word, self.voc_type)
        return img, label_str


class lmdbDataset_real(Dataset):
    def __init__(self, root=None, voc_type='upper', max_len=100, test=False):
        super(lmdbDataset_real, self).__init__()
        self.env = lmdb.open(
            root,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)

        if not self.env:
            print('cannot creat lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get(b'num-samples'))
            self.nSamples = nSamples

        print("We have ", self.nSamples, "samples...")

        self.voc_type = voc_type
        self.max_len = max_len
        self.test = test

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index += 1
        txn = self.env.begin(write=False)
        label_key = b'label-%09d' % index
        word = str(txn.get(label_key).decode())
        img_HR_key = b'image_hr-%09d' % index  # 128*32
        img_lr_key = b'image_lr-%09d' % index  # 64*16
        try:
            img_HR = buf2PIL(txn, img_HR_key, 'RGB')
            img_lr = buf2PIL(txn, img_lr_key, 'RGB')

            # print("img_HR:", img_HR.size, img_lr.size())

        except IOError or len(word) > self.max_len:
            return self[index + 1]
        label_str = str_filt(word, self.voc_type)
        return img_HR, img_lr, label_str


class lmdbDataset_realIC15TextSR(Dataset):
    def __init__(self, root=None, voc_type='upper', max_len=100, test=False):
        super(lmdbDataset_realIC15TextSR, self).__init__()

        # root should be detailed by upper folder of images
        hr_image_dir = os.path.join(root, "HR")
        lr_image_dir = os.path.join(root, "LR")
        anno_dir = os.path.join(root, "ANNOTATION")

        hr_image_list = os.listdir(hr_image_dir)

        self.hr_impath_list = []
        self.lr_impath_list = []
        self.anno_list = []

        print("collect images from:", root)

        mode = "train" if root.split("/")[-2] == "TRAIN" else "test"

        for i in range(len(hr_image_list)):
            hr_impath = os.path.join(hr_image_dir, mode + '-hr-' + str(i+1).rjust(4, '0') + ".pgm")
            lr_impath = os.path.join(lr_image_dir, mode + '-lr-' + str(i+1).rjust(4, '0') + ".pgm")
            anno_path = os.path.join(anno_dir, mode + '-annot-' + str(i+1).rjust(4, '0') + ".txt")

            self.hr_impath_list.append(hr_impath)
            self.lr_impath_list.append(lr_impath)
            self.anno_list.append(anno_path)

        self.nSamples = len(self.anno_list)
        print("Done, we have ", self.nSamples, "samples...")

        self.voc_type = voc_type
        self.max_len = max_len
        self.test = test

    def read_pgm(self, filename, byteorder='>'):
        """Return image data from a raw PGM file as numpy array.

        Format specification: http://netpbm.sourceforge.net/doc/pgm.html

        """
        with open(filename, 'rb') as f:
            buffer = f.read()
        try:
            header, width, height, maxval = re.search(
                b"(^P5\s(?:\s*#.*[\r\n])*"
                b"(\d+)\s(?:\s*#.*[\r\n])*"
                b"(\d+)\s(?:\s*#.*[\r\n])*"
                b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()

            return np.frombuffer(buffer,
                                 dtype='u1' if int(maxval) < 256 else byteorder + 'u2',
                                 count=int(width) * int(height),
                                 offset=len(header)
                                 ).reshape((int(height), int(width)))

        except AttributeError:
            raise ValueError("Not a raw PGM file: '%s'" % filename)


    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):

        idx = index % self.nSamples

        # assert index <= len(self), 'index range error'

        if not os.path.isfile(self.hr_impath_list[idx]):
            print("File not found for", self.hr_impath_list[idx])
            return self[index+1]
        try:
            img_HR_np = self.read_pgm(self.hr_impath_list[idx], byteorder='<')
            img_lr_np = self.read_pgm(self.lr_impath_list[idx], byteorder='<')

            label_str = open(self.anno_list[idx], "r").readlines()[0].replace("\n", "").strip()
            label_str = str_filt(label_str, self.voc_type)
        except ValueError:
            print("File not found for", self.hr_impath_list[idx])
            return self[index + 1]
        # print("annos:", img_HR_np.shape, img_lr_np.shape)

        img_HR = Image.fromarray(cv2.cvtColor(img_HR_np, cv2.COLOR_GRAY2RGB))
        img_lr = Image.fromarray(cv2.cvtColor(img_lr_np, cv2.COLOR_GRAY2RGB))

        return img_HR, img_lr, label_str



class lmdbDataset_realSVT(Dataset):
    def __init__(self, root=None, voc_type='upper', max_len=100, test=False):
        super(lmdbDataset_realSVT, self).__init__()

        # root should be detailed by upper folder of images

        # anno_dir = os.path.join(root, "ANNOTATION")

        split = ("svt_" + "train") if not test else ("svt_" + "test")
        dataset_dir = os.path.join(root, split)
        self.image_dir = os.path.join(dataset_dir, "IMG")
        self.anno_dir = os.path.join(dataset_dir, "label")
        # self.impath_list = os.listdir(image_dir)
        self.anno_list = os.listdir(self.anno_dir)

        # self.impath_list = []
        # self.anno_list = []

        print("collect images from:", root)

        # mode = "train" if root.split("/")[-2] == "TRAIN" else "test"
        self.nSamples = len(self.anno_list)
        print("Done, we have ", self.nSamples, "samples...")

        self.voc_type = voc_type
        self.max_len = max_len
        self.test = test


    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):

        idx = index % self.nSamples

        anno = self.anno_list[index]
        image_path = os.path.join(self.image_dir, anno.split(".")[0] + ".jpg")
        anno_path = os.path.join(self.anno_dir, anno)

        if not os.path.isfile(image_path):
            print("File not found for", image_path)
            return self[index+1]
        try:
            word = open(anno_path, "r").readlines()[0].replace("\n", "")
            img_HR = Image.open(image_path)
            img_lr = img_HR
        except ValueError:
            print("File not found for", image_path)
            return self[index + 1]
        # print("annos:", img_HR_np.shape, img_lr_np.shape)
        label_str = str_filt(word, self.voc_type)

        return img_HR, img_lr, label_str, image_path


class lmdbDataset_realForTest(Dataset):
    def __init__(self, root=None, voc_type='upper', max_len=100, test=False):
        super(lmdbDataset_realForTest, self).__init__()

        # root should be detailed by upper folder of images

        # anno_dir = os.path.join(root, "ANNOTATION")

        # split = ("svt_" + "train") if not test else ("svt_" + "test")
        dataset_dir = root #os.path.join(root, split)
        self.image_dir = os.path.join(dataset_dir)
        # self.anno_dir = os.path.join(dataset_dir, "label")
        # self.impath_list = os.listdir(image_dir)
        # self.anno_list = os.listdir(self.anno_dir)

        self.impath_list = os.listdir(self.image_dir)
        # self.anno_list = []

        print("collect images from:", root)

        # mode = "train" if root.split("/")[-2] == "TRAIN" else "test"
        self.nSamples = len(self.impath_list)
        print("Done, we have ", self.nSamples, "samples...")

        self.voc_type = voc_type
        self.max_len = max_len
        self.test = test


    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):

        idx = index % self.nSamples

        # anno = self.anno_list[index]
        image_path = os.path.join(self.image_dir, self.impath_list[idx])
        # anno_path = os.path.join(self.anno_dir, anno)

        if not os.path.isfile(image_path):
            print("File not found for", image_path)
            return self[index+1]
        try:
            word = "gogogo"
            img_HR = Image.open(image_path)
            img_lr = img_HR
        except ValueError:
            print("File not found for", image_path)
            return self[index + 1]
        # print("annos:", img_HR_np.shape, img_lr_np.shape)
        label_str = str_filt(word, self.voc_type)

        return img_HR, img_lr, label_str# , image_path


class lmdbDataset_realIIIT(Dataset):
    def __init__(self, root=None, voc_type='upper', max_len=100, test=False):
        super(lmdbDataset_realIIIT, self).__init__()

        # root should be detailed by upper folder of images

        # anno_dir = os.path.join(root, "ANNOTATION")

        split = ("iiit5k_" + "train") if not test else ("iiit5k_" + "test")
        dataset_dir = os.path.join(root, split)
        self.image_dir = os.path.join(dataset_dir, "IMG")
        self.anno_dir = os.path.join(dataset_dir, "label")
        # self.impath_list = os.listdir(image_dir)
        self.anno_list = os.listdir(self.anno_dir)

        # self.impath_list = []
        # self.anno_list = []

        print("collect images from:", root)

        # mode = "train" if root.split("/")[-2] == "TRAIN" else "test"
        self.nSamples = len(self.anno_list)
        print("Done, we have ", self.nSamples, "samples...")

        self.voc_type = voc_type
        self.max_len = max_len
        self.test = test


    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):

        idx = index % self.nSamples

        anno = self.anno_list[index]
        image_path = os.path.join(self.image_dir, anno.split(".")[0] + ".jpg")
        anno_path = os.path.join(self.anno_dir, anno)

        if not os.path.isfile(image_path):
            print("File not found for", image_path)
            return self[index+1]
        try:
            word = open(anno_path, "r").readlines()[0].replace("\n", "")
            img_HR = Image.open(image_path)
            img_lr = img_HR
        except ValueError:
            print("File not found for", image_path)
            return self[index + 1]
        # print("annos:", img_HR_np.shape, img_lr_np.shape)
        label_str = str_filt(word, self.voc_type)

        return img_HR, img_lr, label_str, image_path


class lmdbDataset_realBadSet(Dataset):
    def __init__(self, root=None, voc_type='upper', max_len=100, test=False):
        super(lmdbDataset_realBadSet, self).__init__()

        # root should be detailed by upper folder of images

        # anno_dir = os.path.join(root, "ANNOTATION")

        self.imlist = os.listdir(root)
        self.image_dir = root
        # self.impath_list = []
        # self.anno_list = []

        print("collect images from:", root)

        # mode = "train" if root.split("/")[-2] == "TRAIN" else "test"
        self.nSamples = len(self.imlist)
        print("Done, we have ", self.nSamples, "samples...")

        self.voc_type = voc_type
        self.max_len = max_len
        self.test = test


    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):

        idx = index % self.nSamples

        imfile = self.imlist[index]
        image_path = os.path.join(self.image_dir, imfile)

        word = imfile.split("_")[1]

        if not os.path.isfile(image_path):
            print("File not found for", image_path)
            return self[index+1]
        try:
            img_HR = Image.open(image_path)
            img_lr = img_HR
        except ValueError:
            print("File not found for", image_path)
            return self[index + 1]
        # print("annos:", img_HR_np.shape, img_lr_np.shape)
        label_str = str_filt(word, self.voc_type)

        return img_HR, img_lr, label_str, image_path


class lmdbDataset_realIC15(Dataset):
    def __init__(self, root=None, voc_type='upper', max_len=100, test=False):
        super(lmdbDataset_realIC15, self).__init__()

        self.env = lmdb.open(
            root,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)

        if not self.env:
            print('cannot creat lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get(b'num-samples'))
            self.nSamples = nSamples
        self.voc_type = voc_type
        self.max_len = max_len
        self.test = test

        # print("ROOT:", root)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index += 1
        txn = self.env.begin(write=False)
        label_key = b'label-%09d' % index
        word = str(txn.get(label_key).decode())
        img_key = b'image-%09d' % index  # 128*32
        # img_lr_key = b'image_lr-%09d' % index  # 64*16
        try:
            img_HR = buf2PIL(txn, img_key, 'RGB')

            img_lr = img_HR
            # print("img:", img_HR.size, word)
            # img_lr = buf2PIL(txn, img_lr_key, 'RGB')
        except IOError or len(word) > self.max_len:
            return self[index + 1]
        label_str = str_filt(word, self.voc_type)
        return img_HR, img_lr, label_str, str(img_key)


class lmdbDataset_realCOCOText(Dataset):
    def __init__(self, root=None, voc_type='upper', max_len=100, test=False):
        super(lmdbDataset_realCOCOText, self).__init__()

        if test:
            gt_file = "val_words_gt.txt"
            im_dir = "val_words"
        else:
            gt_file = "train_words_gt.txt"
            im_dir = "train_words"

        self.image_dir = os.path.join(root, im_dir)
        self.gt_file = os.path.join(root, gt_file)

        self.gtlist = open(self.gt_file, "r").readlines()

        if test:
            self.gtlist = self.gtlist[:3000]

        self.nSamples = len(self.gtlist)

        self.voc_type = voc_type
        self.max_len = max_len
        self.test = test

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        # index += 1

        gt_anno = self.gtlist[index].replace("\n", "")
        if len(gt_anno.split(",")) < 2:
            return self[index + 1]
        img_id, label_str = gt_anno.split(",")[:2]
        impath = os.path.join(self.image_dir, img_id + ".jpg")

        try:
            img_HR = Image.open(impath)
            img_lr = img_HR
            # print("img:", img_HR.size, word)
            # img_lr = buf2PIL(txn, img_lr_key, 'RGB')
        except IOError or len(label_str) > self.max_len:
            return self[index + 1]
        label_str = str_filt(label_str, self.voc_type)
        return img_HR, img_lr, label_str, impath


class lmdbDatasetWithW2V_real(Dataset):
    def __init__(
                     self,
                     root=None,
                     voc_type='upper',
                     max_len=100,
                     test=False,
                     w2v_lexicons="cc.en.300.bin"
                 ):
        super(lmdbDatasetWithW2V_real, self).__init__()
        self.env = lmdb.open(
            root,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)

        if not self.env:
            print('cannot creat lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get(b'num-samples'))
            self.nSamples = nSamples
        self.voc_type = voc_type
        self.max_len = max_len
        self.test = test

        # self.w2v_lexicon = FastText(w2v_lexicons)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index += 1
        txn = self.env.begin(write=False)
        label_key = b'label-%09d' % index
        word = str(txn.get(label_key).decode())
        img_HR_key = b'image_hr-%09d' % index  # 128*32
        img_lr_key = b'image_lr-%09d' % index  # 64*16
        try:
            img_HR = buf2PIL(txn, img_HR_key, 'RGB')
            img_lr = buf2PIL(txn, img_lr_key, 'RGB')
        except IOError or len(word) > self.max_len:
            return self[index + 1]
        label_str = str_filt(word, self.voc_type)

        # print("HR, LR:", img_HR.size, img_lr.size)

        w2v = None# self.w2v_lexicon.get_numpy_vector(label_str.lower())

        return img_HR, img_lr, label_str, w2v



class resizeNormalize(object):
    def __init__(self, size, mask=False, interpolation=Image.BICUBIC):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()
        self.mask = mask

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img_tensor = self.toTensor(img)
        if self.mask:
            mask = img.convert('L')
            thres = np.array(mask).mean()
            mask = mask.point(lambda x: 0 if x > thres else 255)
            mask = self.toTensor(mask)
            img_tensor = torch.cat((img_tensor, mask), 0)

        return img_tensor


class Normalize(object):
    def __init__(self, size, mask=False, interpolation=Image.BICUBIC):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()
        self.mask = mask

    def __call__(self, img):
        # img = img.resize(self.size, self.interpolation)
        img_tensor = self.toTensor(img)
        if self.mask:
            mask = img.convert('L')
            thres = np.array(mask).mean()
            mask = mask.point(lambda x: 0 if x > thres else 255)
            mask = self.toTensor(mask)
            img_tensor = torch.cat((img_tensor, mask), 0)

        return img_tensor



class resizeNormalizeRandomCrop(object):
    def __init__(self, size, mask=False, interpolation=Image.BICUBIC):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()
        self.mask = mask

    def __call__(self, img, interval=None):

        w, h = img.size

        if w < 32 or not interval is None:
            img = img.resize(self.size, self.interpolation)
            img_tensor = self.toTensor(img)
        else:
            np_img = np.array(img)
            h, w = np_img.shape[:2]
            np_img_crop = np_img[:, int(w * interval[0]):int(w * interval[1])]
            # print("size:", self.size, np_img_crop.shape, np_img.shape, interval)
            img = Image.fromarray(np_img_crop)
            img = img.resize(self.size, self.interpolation)
            img_tensor = self.toTensor(img)

        if self.mask:
            mask = img.convert('L')
            thres = np.array(mask).mean()
            mask = mask.point(lambda x: 0 if x > thres else 255)
            mask = self.toTensor(mask)
            img_tensor = torch.cat((img_tensor, mask), 0)

        return img_tensor


class resizeNormalizeKeepRatio(object):
    def __init__(self, size, mask=False, interpolation=Image.BICUBIC):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()
        self.mask = mask

    def __call__(self, img, label_str):
        o_w, o_h = img.size

        ratio = o_w / float(o_h)
        re_h = self.size[1]
        re_w = int(re_h * ratio)
        if re_w > self.size[0]:
            img = img.resize(self.size, self.interpolation)
            img_tensor = self.toTensor(img).float()
        else:
            img = img.resize((re_w, re_h), self.interpolation)
            img_np = np.array(img)
            # if len(label_str) > 4:
            #     print("img_np:", img_np.shape)

            shift_w = int((self.size[0] - img_np.shape[1]) / 2)
            re_img = np.zeros((self.size[1], self.size[0], img_np.shape[-1]))
            re_img[:, shift_w:img_np.shape[1]+shift_w] = img_np

            re_img = Image.fromarray(re_img.astype(np.uint8))

            img_tensor = self.toTensor(re_img).float()

            if o_h / o_w < 0.5 and len(label_str) > 4:
                # cv2.imwrite("mask_h_" + label_str + ".jpg", re_mask.astype(np.uint8))
                # cv2.imwrite("img_h_" + label_str + ".jpg", np.array(re_img))
                # print("img_np_h:", o_h, o_w, img_np.shape, label_str)
                pass

        if self.mask:
            mask = img.convert('L')
            thres = np.array(mask).mean()
            mask = mask.point(lambda x: 0 if x > thres else 255)
            if re_w > self.size[0]:
                # img = img.resize(self.size, self.interpolation)

                re_mask_cpy = np.ones((mask.size[1], mask.size[0]))

                mask = self.toTensor(mask)
                img_tensor = torch.cat((img_tensor, mask), 0).float()
            else:
                mask = np.array(mask)
                mask = cv2.resize(mask, (re_w, re_h), cv2.INTER_NEAREST)
                shift_w = int((self.size[0] - mask.shape[1]) / 2)

                # print("resize mask:", mask.shape)

                re_mask = np.zeros((self.size[1], self.size[0]))

                re_mask_cpy = re_mask.copy()
                re_mask_cpy[:, shift_w:mask.shape[1] + shift_w] = np.ones(mask.shape)

                re_mask[:, shift_w:mask.shape[1] + shift_w] = mask
                '''
                if o_h / o_w > 2 and len(label_str) > 4:
                    cv2.imwrite("mask_" + label_str + ".jpg", re_mask.astype(np.uint8))
                    cv2.imwrite("img_" + label_str + ".jpg", re_img.astype(np.uint8))
                    print("img_np:", o_h, o_w, img_np.shape, label_str)

                if o_h / o_w < 0.5 and len(label_str) > 4:
                    cv2.imwrite("mask_h_" + label_str + ".jpg", re_mask.astype(np.uint8))
                    cv2.imwrite("img_h_" + label_str + ".jpg", re_img.astype(np.uint8))
                    print("img_np_h:", o_h, o_w, img_np.shape, label_str)
                '''
                re_mask = self.toTensor(re_mask).float()
                img_tensor = torch.cat((img_tensor, re_mask), 0)

        return img_tensor, torch.tensor(cv2.resize(re_mask_cpy, (self.size[0] * 2, self.size[1] * 2), cv2.INTER_NEAREST)).float()


class lmdbDataset_mix(Dataset):
    def __init__(self, root=None, voc_type='upper', max_len=100, test=False):
        super(lmdbDataset_mix, self).__init__()
        self.env = lmdb.open(
            root,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)

        if not self.env:
            print('cannot creat lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get(b'num-samples'))
            self.nSamples = nSamples
        self.voc_type = voc_type
        self.max_len = max_len
        self.test = test

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index += 1
        txn = self.env.begin(write=False)
        label_key = b'label-%09d' % index
        word = str(txn.get(label_key).decode())
        if self.test:
            try:
                img_HR = buf2PIL(txn, b'image_hr-%09d' % index, 'RGB')
                img_lr = buf2PIL(txn, b'image_lr-%09d' % index, 'RGB')
            except:
                img_HR = buf2PIL(txn, b'image-%09d' % index, 'RGB')
                img_lr = img_HR

        else:
            img_HR = buf2PIL(txn, b'image_hr-%09d' % index, 'RGB')
            if random.uniform(0, 1) < 0.5:
                img_lr = buf2PIL(txn, b'image_lr-%09d' % index, 'RGB')
            else:
                img_lr = img_HR

        label_str = str_filt(word, self.voc_type)
        return img_HR, img_lr, label_str


class lmdbDatasetWithMask_real(Dataset):
    def __init__(self, root=None, voc_type='upper', max_len=100, test=False):
        super(lmdbDatasetWithMask_real, self).__init__()
        self.env = lmdb.open(
            root,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)

        if not self.env:
            print('cannot creat lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get(b'num-samples'))
            self.nSamples = nSamples
        self.voc_type = voc_type
        self.max_len = max_len
        self.test = test

    def __len__(self):
        return self.nSamples

    def get_mask(self, image):

        img_hr = np.array(image)
        img_hr_gray = cv2.cvtColor(img_hr, cv2.COLOR_BGR2GRAY)

        kernel = np.ones((5, 5), np.uint8)
        hr_canny = cv2.Canny(img_hr_gray, 20, 150)
        hr_canny = cv2.dilate(hr_canny, kernel, iterations=1)
        hr_canny = cv2.GaussianBlur(hr_canny, (5, 5), 1)
        weighted_mask = 0.4 + (hr_canny / 255.0) * 0.5

        return weighted_mask

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index += 1
        txn = self.env.begin(write=False)
        label_key = b'label-%09d' % index
        word = str(txn.get(label_key).decode())
        img_HR_key = b'image_hr-%09d' % index  # 128*32
        img_lr_key = b'image_lr-%09d' % index  # 64*16
        try:
            img_HR = buf2PIL(txn, img_HR_key, 'RGB')
            img_lr = buf2PIL(txn, img_lr_key, 'RGB')
        except IOError or len(word) > self.max_len:
            return self[index + 1]
        label_str = str_filt(word, self.voc_type)

        weighted_mask = self.get_mask(img_HR)

        return img_HR, img_lr, label_str, weighted_mask



class randomSequentialSampler(sampler.Sampler):

    def __init__(self, data_source, batch_size):
        self.num_samples = len(data_source)
        self.batch_size = batch_size

    def __iter__(self):
        n_batch = len(self) // self.batch_size
        tail = len(self) % self.batch_size
        index = torch.LongTensor(len(self)).fill_(0)
        for i in range(n_batch):
            random_start = random.randint(0, len(self) - self.batch_size)
            batch_index = random_start + torch.arange(0, self.batch_size)
            index[i * self.batch_size:(i + 1) * self.batch_size] = batch_index
        # deal with tail
        if tail:
            random_start = random.randint(0, len(self) - self.batch_size)
            tail_index = random_start + torch.arange(0, tail)
            index[(i + 1) * self.batch_size:] = tail_index

        return iter(index)

    def __len__(self):
        return self.num_samples


class alignCollate_syn(object):
    def __init__(self, imgH=64,
                 imgW=256,
                 down_sample_scale=4,
                 keep_ratio=False,
                 min_ratio=1,
                 mask=False,
                 alphabet=53,
                 train=True
                 ):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio = keep_ratio
        self.min_ratio = min_ratio
        self.down_sample_scale = down_sample_scale
        self.mask = mask
        self.alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"
        self.d2a = "-" + self.alphabet
        self.alsize = len(self.d2a)
        self.a2d = {}
        cnt = 0
        for ch in self.d2a:
            self.a2d[ch] = cnt
            cnt += 1

        imgH = self.imgH
        imgW = self.imgW

        self.transform = resizeNormalize((imgW, imgH), self.mask)
        self.transform2 = resizeNormalize((imgW // self.down_sample_scale, imgH // self.down_sample_scale), self.mask)

        self.train = train

    def degradation(self, img_L):
        # degradation process, blur + bicubic downsampling + Gaussian noise
        # if need_degradation:
        # img_L = util.modcrop(img_L, sf)
        img_L = np.array(img_L)
        # print("img_L_before:", img_L.shape, np.unique(img_L))
        img_L = sr.srmd_degradation(img_L, kernel)

        noise_level_img = 0.
        if not self.train:
            np.random.seed(seed=0)  # for reproducibility
        # print("unique:", np.unique(img_L))
        img_L = img_L + np.random.normal(0, noise_level_img, img_L.shape)

        # print("img_L_after:", img_L_beore.shape, img_L.shape, np.unique(img_L))

        return Image.fromarray(img_L.astype(np.uint8))

    def __call__(self, batch):
        images, _, label_strs, identity = zip(*batch)

        # [self.degradation(image) for image in images]
        # images_hr = images
        '''
        images_lr = [image.resize(
            (image.size[0] // self.down_sample_scale, image.size[1] // self.down_sample_scale),
            Image.BICUBIC) for image in images]

        if self.train:
            if random.random() > 1.5:
                images_hr = [image.resize(
                (image.size[0]//self.down_sample_scale, image.size[1]//self.down_sample_scale),
                Image.BICUBIC) for image in images]
            else:
                images_hr = images
        else:
            images_hr = images
            #[image.resize(
            #    (image.size[0] // self.down_sample_scale, image.size[1] // self.down_sample_scale),
            #    Image.BICUBIC) for image in images]
        '''
        # images_hr = [self.degradation(image) for image in images]
        images_lr = images_hr = images
        # images_lr = [image.resize(
        #    (image.size[0] // 2, image.size[1] // 2),
        #    Image.BICUBIC) for image in images]

        images_hr = [self.transform(image) for image in images_hr]
        images_hr = torch.cat([t.unsqueeze(0) for t in images_hr], 0)

        images_lr = [image.resize(
        (image.size[0]//self.down_sample_scale, image.size[1]//self.down_sample_scale),
        Image.BICUBIC) for image in images]

        # images_lr = [self.degradation(image) for image in images]
        images_lr = [self.transform2(image) for image in images_lr]
        images_lr = torch.cat([t.unsqueeze(0) for t in images_lr], 0)

        return images_hr, images_lr, label_strs, identity



class alignCollate_syn_random_reso(object):
    def __init__(self, imgH=64,
                 imgW=256,
                 down_sample_scale=4,
                 keep_ratio=False,
                 min_ratio=1,
                 mask=False,
                 alphabet=53,
                 train=True
                 ):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio = keep_ratio
        self.min_ratio = min_ratio
        self.down_sample_scale = down_sample_scale
        self.mask = mask
        self.alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"
        self.d2a = "-" + self.alphabet
        self.alsize = len(self.d2a)
        self.a2d = {}
        cnt = 0
        for ch in self.d2a:
            self.a2d[ch] = cnt
            cnt += 1

        imgH = self.imgH
        imgW = self.imgW

        self.transform = Normalize((imgW, imgH), self.mask)
        self.transform2 = Normalize((imgW // self.down_sample_scale, imgH // self.down_sample_scale), self.mask)

        self.train = train

    def degradation(self, img_L):
        # degradation process, blur + bicubic downsampling + Gaussian noise
        # if need_degradation:
        # img_L = util.modcrop(img_L, sf)
        img_L = np.array(img_L)
        # print("img_L_before:", img_L.shape, np.unique(img_L))
        img_L = sr.srmd_degradation(img_L, kernel)

        noise_level_img = 0.
        if not self.train:
            np.random.seed(seed=0)  # for reproducibility
        # print("unique:", np.unique(img_L))
        img_L = img_L + np.random.normal(0, noise_level_img, img_L.shape)

        # print("img_L_after:", img_L_beore.shape, img_L.shape, np.unique(img_L))

        return Image.fromarray(img_L.astype(np.uint8))

    def __call__(self, batch):
        images, _, label_strs, identity = zip(*batch)

        # [self.degradation(image) for image in images]
        # images_hr = images
        '''
        images_lr = [image.resize(
            (image.size[0] // self.down_sample_scale, image.size[1] // self.down_sample_scale),
            Image.BICUBIC) for image in images]

        if self.train:
            if random.random() > 1.5:
                images_hr = [image.resize(
                (image.size[0]//self.down_sample_scale, image.size[1]//self.down_sample_scale),
                Image.BICUBIC) for image in images]
            else:
                images_hr = images
        else:
            images_hr = images
            #[image.resize(
            #    (image.size[0] // self.down_sample_scale, image.size[1] // self.down_sample_scale),
            #    Image.BICUBIC) for image in images]
        '''
        # images_hr = [self.degradation(image) for image in images]
        images_lr = images_hr = images
        # images_lr = [image.resize(
        #    (image.size[0] // 2, image.size[1] // 2),
        #    Image.BICUBIC) for image in images]

        images_hr = [img.resize((img.size[0] * 2 , img.size[1] * 2), Image.BICUBIC) for img in images_hr]
        images_hr = [self.transform(image).unsqueeze(0) for image in images_hr]

        # images_hr = torch.cat([t.unsqueeze(0) for t in images_hr], 0)

        # images_lr = [image.resize(
        # (image.size[0]//self.down_sample_scale, image.size[1]//self.down_sample_scale),
        # Image.BICUBIC) for image in images]

        # images_lr = [self.degradation(image) for image in images]
        images_lr = [self.transform2(image).unsqueeze(0) for image in images_lr]
        # images_lr = torch.cat([t.unsqueeze(0) for t in images_lr], 0)

        return images_hr, images_lr, label_strs, identity


class alignCollate_syn_withcrop(object):
    def __init__(self, imgH=64,
                 imgW=256,
                 down_sample_scale=4,
                 keep_ratio=False,
                 min_ratio=1,
                 mask=False,
                 alphabet=53,
                 train=True
                 ):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio = keep_ratio
        self.min_ratio = min_ratio
        self.down_sample_scale = down_sample_scale
        self.mask = mask
        self.alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"
        self.d2a = "-" + self.alphabet
        self.alsize = len(self.d2a)
        self.a2d = {}
        cnt = 0
        for ch in self.d2a:
            self.a2d[ch] = cnt
            cnt += 1

        imgH = self.imgH
        imgW = self.imgW

        self.transform = resizeNormalizeRandomCrop((imgW, imgH), self.mask)
        self.transform2 = resizeNormalizeRandomCrop((imgW // self.down_sample_scale, imgH // self.down_sample_scale),
                                                    self.mask)

    def __call__(self, batch):
        images, label_strs = zip(*batch)

        images_hr = [self.transform(image) for image in images]
        images_hr = torch.cat([t.unsqueeze(0) for t in images_hr], 0)

        images_lr = [image.resize((image.size[0]//self.down_sample_scale, image.size[1]//self.down_sample_scale), Image.BICUBIC) for image in images]
        images_lr = [self.transform2(image) for image in images_lr]
        images_lr = torch.cat([t.unsqueeze(0) for t in images_lr], 0)


        return images_hr, images_lr, label_strs



class alignCollate_real(alignCollate_syn):
    def __call__(self, batch):
        images_HR, images_lr, label_strs, _ = zip(*batch)
        imgH = self.imgH
        imgW = self.imgW
        transform = resizeNormalize((imgW, imgH), self.mask)
        transform2 = resizeNormalize((imgW // self.down_sample_scale, imgH // self.down_sample_scale), self.mask)
        images_HR = [transform(image) for image in images_HR]
        images_HR = torch.cat([t.unsqueeze(0) for t in images_HR], 0)

        images_lr = [transform2(image) for image in images_lr]
        images_lr = torch.cat([t.unsqueeze(0) for t in images_lr], 0)

        return images_HR, images_lr, label_strs, _


class alignCollate_realWTL(alignCollate_syn):
    def __call__(self, batch):
        images_HR, images_lr, label_strs = zip(*batch)
        imgH = self.imgH
        imgW = self.imgW
        # transform = resizeNormalize((imgW, imgH), self.mask)
        # transform2 = resizeNormalize((imgW // self.down_sample_scale, imgH // self.down_sample_scale), self.mask)
        images_HR = [self.transform(image) for image in images_HR]
        images_HR = torch.cat([t.unsqueeze(0) for t in images_HR], 0)

        images_lr = [self.transform2(image) for image in images_lr]
        images_lr = torch.cat([t.unsqueeze(0) for t in images_lr], 0)

        max_len = 0

        label_batches = []

        for word in label_strs:
            word = word.lower()
            # Complement

            if len(word) > 4:
                word = [ch for ch in word]
                word[2] = "e"
                word = "".join(word)

            if len(word) <= 1:
                pass
            elif len(word) < 26 and len(word) > 1:
                inter_com = 26 - len(word)
                # padding = int(inter_com / (len(word) - 1))
                # new_word = word[0]
                # for i in range(len(word) - 1):
                #    new_word += "-" * padding + word[i+1]

                # word = new_word
                pass
            else:
                word = word[:26]

            label_list = [self.a2d[ch] for ch in word if ch in self.a2d]

            # shifting:
            # if len(label_list) > 2:
                #     if label_list[-1] > 0 and label_list[-1] < self.alsize - 1:
            #     label_list[-1] = 0

            word_len = len(word)
            if word_len > max_len:
                max_len = word_len
            # print("label_list:", word, label_list)
            labels = torch.tensor(label_list)[:, None].long()
            label_vecs = torch.zeros((labels.shape[0], self.alsize))
            # print("labels:", labels)
            if labels.shape[0] > 0:
                label_batches.append(label_vecs.scatter_(-1, labels, 1))
            else:
                label_batches.append(label_vecs)
        label_rebatches = torch.zeros((len(label_strs), max_len, self.alsize))

        for idx in range(len(label_strs)):
            label_rebatches[idx][:label_batches[idx].shape[0]] = label_batches[idx]

        label_rebatches = label_rebatches.unsqueeze(1).float().permute(0, 3, 1, 2)
        # noise = (torch.rand(label_rebatches.shape) - 0.5) * 0.2

        # label_rebatches#  += noise

        return images_HR, images_lr, label_strs, label_rebatches


class alignCollate_realWTLAMask(alignCollate_syn):

    def get_mask(self, image):
        img_hr = np.transpose(image.data.numpy() * 255, (1, 2, 0))
        img_hr_gray = cv2.cvtColor(img_hr[..., :3].astype(np.uint8), cv2.COLOR_BGR2GRAY)
        # print("img_hr_gray: ", np.unique(img_hr_gray), img_hr_gray.shape)
        kernel = np.ones((5, 5), np.uint8)
        hr_canny = cv2.Canny(img_hr_gray, 20, 150)
        hr_canny = cv2.dilate(hr_canny, kernel, iterations=1)
        hr_canny = cv2.GaussianBlur(hr_canny, (5, 5), 1)
        weighted_mask = 0.4 + (hr_canny / 255.0) * 0.6
        return torch.tensor(weighted_mask).float().unsqueeze(0)

    def __call__(self, batch):
        images_HR, images_lr, label_strs = zip(*batch)
        imgH = self.imgH
        imgW = self.imgW
        # transform = resizeNormalize((imgW, imgH), self.mask)
        # transform2 = resizeNormalize((imgW // self.down_sample_scale, imgH // self.down_sample_scale), self.mask)
        images_HR = [self.transform(image) for image in images_HR]
        images_HR = torch.cat([t.unsqueeze(0) for t in images_HR], 0)

        images_lr = [self.transform2(image) for image in images_lr]
        images_lr = torch.cat([t.unsqueeze(0) for t in images_lr], 0)

        # weighted_masks = [self.get_mask(image_HR) for image_HR in images_HR]
        # weighted_masks = torch.cat([t.unsqueeze(0) for t in weighted_masks], 0)

        # print("weighted_masks:", weighted_masks.shape, np.unique(weighted_masks))
        max_len = 0

        label_batches = []
        weighted_masks = []
        weighted_tics = []

        for word in label_strs:
            word = word.lower()
            # Complement

            if len(word) > 4:
                # word = [ch for ch in word]
                # word[2] = "e"
                # word = "".join(word)
                pass
            if len(word) <= 1:
                pass
            elif len(word) < 15 and len(word) > 1:
                # inter_com = 26 - len(word)
                # padding = int(inter_com / (len(word) - 1))
                # new_word = word[0]
                # for i in range(len(word) - 1):
                #     new_word += "-" * padding + word[i+1]

                # word = new_word
                pass
            else:
                word = word[:15]

            label_list = [self.a2d[ch] for ch in word if ch in self.a2d]

            if len(label_list) <= 0:
                # blank label
                weighted_masks.append(0)
            else:
                weighted_masks.extend(label_list)

            word_len = len(word)
            if word_len > max_len:
                max_len = word_len
            # print("label_list:", word, label_list)
            labels = torch.tensor(label_list)[:, None].long()

            # print("labels:", labels)

            if labels.shape[0] > 0:
                label_vecs = torch.zeros((labels.shape[0], self.alsize))
                # print(label_vecs.scatter_(-1, labels, 1))
                label_batches.append(label_vecs.scatter_(-1, labels, 1))
                weighted_tics.append(1)
            else:
                label_vecs = torch.zeros((1, self.alsize))
                # Assign a blank label
                label_vecs[0, 0] = 1.
                label_batches.append(label_vecs)
                weighted_tics.append(0)
        label_rebatches = torch.zeros((len(label_strs), max_len, self.alsize))

        for idx in range(len(label_strs)):
            label_rebatches[idx][:label_batches[idx].shape[0]] = label_batches[idx]

        label_rebatches = label_rebatches.unsqueeze(1).float().permute(0, 3, 1, 2)
        # noise = (torch.rand(label_rebatches.shape) - 0.5) * 0.2

        # label_rebatches += noise

        # print("images_HR:", images_HR.shape, images_lr.shape)

        return images_HR, images_lr, label_strs, label_rebatches, torch.tensor(weighted_masks).long(), torch.tensor(weighted_tics)


import random
class alignCollate_realWTL_withcrop(alignCollate_syn_withcrop):
    def __call__(self, batch):
        images_HR, images_lr, label_strs = zip(*batch)
        imgH = self.imgH
        imgW = self.imgW

        # transform = resizeNormalize((imgW, imgH), self.mask)
        # transform2 = resizeNormalize((imgW // self.down_sample_scale, imgH // self.down_sample_scale), self.mask)

        HR_list = []
        LR_list = []

        for i in range(len(images_HR)):

            shift_proportion = 0.4 * random.random()
            l_shift = random.random() * shift_proportion
            r_shift = shift_proportion - l_shift
            interval = [l_shift, 1 - r_shift]
            HR_list.append(self.transform(images_HR[i], interval))
            LR_list.append(self.transform2(images_lr[i], interval))

        images_HR = torch.cat([t.unsqueeze(0) for t in HR_list], 0)
        images_lr = torch.cat([t.unsqueeze(0) for t in LR_list], 0)

        # images_HR = [self.transform(image) for image in images_HR]
        # images_HR = torch.cat([t.unsqueeze(0) for t in images_HR], 0)

        # images_lr = [self.transform2(image) for image in images_lr]
        # images_lr = torch.cat([t.unsqueeze(0) for t in images_lr], 0)

        max_len = 0

        label_batches = []

        for word in label_strs:
            word = word.lower()
            # Complement

            if len(word) > 4:
                word = [ch for ch in word]
                word[2] = "e"
                word = "".join(word)

            if len(word) <= 1:
                pass
            elif len(word) < 26 and len(word) > 1:
                inter_com = 26 - len(word)
                padding = int(inter_com / (len(word) - 1))
                new_word = word[0]
                for i in range(len(word) - 1):
                    new_word += "-" * padding + word[i+1]

                word = new_word
            else:
                word = word[:26]

            label_list = [self.a2d[ch] for ch in word if ch in self.a2d]

            # shifting:
            # if len(label_list) > 2:
                #     if label_list[-1] > 0 and label_list[-1] < self.alsize - 1:
            #     label_list[-1] = 0

            word_len = len(word)
            if word_len > max_len:
                max_len = word_len
            # print("label_list:", word, label_list)
            labels = torch.tensor(label_list)[:, None].long()
            label_vecs = torch.zeros((labels.shape[0], self.alsize))
            # print("labels:", labels)
            if labels.shape[0] > 0:
                label_batches.append(label_vecs.scatter_(-1, labels, 1))
            else:
                label_batches.append(label_vecs)
        label_rebatches = torch.zeros((len(label_strs), max_len, self.alsize))

        for idx in range(len(label_strs)):
            label_rebatches[idx][:label_batches[idx].shape[0]] = label_batches[idx]

        label_rebatches = label_rebatches.unsqueeze(1).float().permute(0, 3, 1, 2)
        noise = (torch.rand(label_rebatches.shape) - 0.5) * 0.2

        label_rebatches += noise

        return images_HR, images_lr, label_strs, label_rebatches


class alignCollateW2V_real(alignCollate_syn):
    def __call__(self, batch):
        images_HR, images_lr, label_strs, w2vs = zip(*batch)
        imgH = self.imgH
        imgW = self.imgW
        transform = resizeNormalize((imgW, imgH), self.mask)
        transform2 = resizeNormalize((imgW // self.down_sample_scale, imgH // self.down_sample_scale), self.mask)

        image_masks = []
        image_lrs = []

        for i in range(len(images_lr)):
            image_lr = transform2(images_lr[i], label_strs[i])
            image_lrs.append(image_lr)
            # image_masks.append(image_mask)

        # images_lr = [transform2(images_lr[i], label_strs[i])[0] for i in range(len(images_lr))]
        images_lr = torch.cat([t.unsqueeze(0) for t in image_lrs], 0)
        # image_masks = torch.cat([t.unsqueeze(0) for t in image_masks], 0)

        images_HR = [transform(images_HR[i], label_strs[i]) for i in range(len(images_HR))]
        images_HR = torch.cat([t.unsqueeze(0) for t in images_HR], 0)

        # print("Align:", type(w2vs), len(w2vs))
        # w2v_tensors = torch.cat([torch.tensor(w2v)[None, ...] for w2v in w2vs], 0).float()
        # print("Align:", type(w2vs), len(w2vs), w2v_tensors.shape)
        w2v_tensors = None

        # print("image_HR:", images_HR.shape, images_lr.shape)

        return images_HR, images_lr, label_strs, w2v_tensors # , image_masks


class alignCollatec2f_real(alignCollate_syn):
    def __call__(self, batch):
        images_HR, images_lr, label_strs = zip(*batch)

        # print("images_HR:", images_HR[0], images_lr[0])

        image_MX = []

        for i in range(len(images_HR)):
            HR_i = np.array(images_HR[i]).astype(np.float32)
            LR_i = np.array(images_lr[i]).astype(np.float32)

            image_MX.append(Image.fromarray(((HR_i + LR_i) / 2.0).astype(np.uint8)))

            # print("unique:", np.unique(HR_i))
            # print("unique:", np.unique(LR_i))

        imgH = self.imgH
        imgW = self.imgW
        transform = resizeNormalize((imgW, imgH), self.mask)
        transform2 = resizeNormalize((imgW // self.down_sample_scale, imgH // self.down_sample_scale), self.mask)
        images_HR = [transform(image) for image in images_HR]
        images_HR = torch.cat([t.unsqueeze(0) for t in images_HR], 0)

        images_lr = [transform2(image) for image in images_lr]
        images_lr = torch.cat([t.unsqueeze(0) for t in images_lr], 0)

        images_MX = [transform2(image) for image in image_MX]
        images_MX = torch.cat([t.unsqueeze(0) for t in images_MX], 0)

        # print("Align:", type(w2vs), len(w2vs))
        # w2v_tensors = torch.cat([torch.tensor(w2v)[None, ...] for w2v in w2vs], 0).float()
        # print("Align:", type(w2vs), len(w2vs), w2v_tensors.shape)

        return images_HR, images_lr, label_strs, images_MX


class ConcatDataset(Dataset):
    """
    Dataset to concatenate multiple datasets.
    Purpose: useful to assemble different existing datasets, possibly
    large-scale datasets as the concatenation operation is done in an
    on-the-fly manner.
    Arguments:
        datasets (sequence): List of datasets to be concatenated
    """

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets):
        super(ConcatDataset, self).__init__()
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.datasets = list(datasets)
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    @property
    def cummulative_sizes(self):
        warnings.warn("cummulative_sizes attribute is renamed to "
                      "cumulative_sizes", DeprecationWarning, stacklevel=2)
        return self.cumulative_sizes


if __name__ == '__main__':
    # embed(header='dataset.py')

    # coding=utf-8
    # import cv2
    # import numpy as np

    root_path = "../../hard_space1/mjq/ic15_4468_train_lmdb/"

    data_annos = lmdbDataset_realIC15(root_path)
    nsamples = data_annos.nSamples

    save_dir = "canny/"
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    """
    for i in range(100):
        img_hr, img_lr, label_str = data_annos[i]

        img_hr = np.array(img_hr)
        img_lr = np.array(img_lr)

        img_hr_gray = cv2.cvtColor(img_hr, cv2.COLOR_BGR2GRAY)
        img_lr_gray = cv2.cvtColor(img_lr, cv2.COLOR_BGR2GRAY)

        # img = cv2.GaussianBlur(img, (3, 3), 0)

        kernel = np.ones((5, 5), np.uint8)
        # erosion = cv2.erode(img, kernel, iterations=1)

        #kernel_w = cv2.getGaussianKernel(5, 0.8)
        #kernel_h = cv2.getGaussianKernel(5, 0.8)
        #kernel_w = np.transpose(kernel_w, (1, 0))
        #kernel = kernel_w * kernel_h

        hr_canny = cv2.Canny(img_hr_gray, 20, 150)
        lr_canny = cv2.Canny(img_lr_gray, 20, 150)

        hr_canny = cv2.dilate(hr_canny, kernel, iterations=1)
        lr_canny = cv2.dilate(lr_canny, kernel, iterations=1)

        hr_canny = cv2.GaussianBlur(hr_canny, (5, 5), 1)
        lr_canny = cv2.GaussianBlur(lr_canny, (5, 5), 1)

        pub_w = max(hr_canny.shape[1], lr_canny.shape[1])
        pub_h = hr_canny.shape[0] + lr_canny.shape[0] + 10

        pub_img = np.zeros((pub_h, pub_w)).astype(np.uint8)
        pub_img[:lr_canny.shape[0], :lr_canny.shape[1]] = lr_canny
        pub_img[lr_canny.shape[0] + 5:lr_canny.shape[0] + 5 + hr_canny.shape[0], :hr_canny.shape[1]] = hr_canny

        print("kernel:", kernel.shape, np.unique(kernel), np.unique(pub_img))

        cv2.imwrite(os.path.join(save_dir, 'Canny' + str(i) + '.jpg'), pub_img)

        # cv2.imshow('Canny', canny)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    """

    min_area = 999999999999
    max_area = 0

    for i in range(nsamples):
        img_hr, img_lr, label_str, _ = data_annos[i]

        if max_area < img_hr.size[0] * img_hr.size[1]:
            max_area = img_hr.size[0] * img_hr.size[1]
            print("max_area:", max_area, img_hr.size)

        if min_area > img_hr.size[0] * img_hr.size[1]:
            min_area = img_hr.size[0] * img_hr.size[1]
            print("min_area:", min_area, img_hr.size)