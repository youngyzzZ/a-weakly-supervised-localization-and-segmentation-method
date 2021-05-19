import datetime
import shutil
import os
import json
from collections import OrderedDict

import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import sys
import platform
import torch
import torch.nn as nn
import torchvision.transforms as transforms
# from openpyxl import load_workbook
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets



# source: source file path
# target:target file path
def copy(source, target):
    if not os.path.exists(source):
        raise RuntimeError('file %s does not exists!' % source)
    shutil.copyfile(source, target)


def move(source, target):
    if not os.path.exists(source):
        raise RuntimeError('source file does not exists!')
    if os.path.exists(target):
        raise RuntimeError('target file has existed!')
    shutil.move(source, target)


# center_crop image
def center_crop(path, new_width, new_height):
    image = Image
    width, height = image.size

    # resize to (224,224) directly if the new height or new width is larger(i.e. enlarge not crop)
    if width < new_width or height < new_height:
        print(path)
        return image.resize((new_width, new_height))

    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2

    return image.crop((left, top, right, bottom))


# del all file
def clear(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        os.mkdir(path)
    else:
        os.mkdir(path)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# write json(dict) to txt file
def write(dic, path):
    with open(path, 'w+') as f:
        f.write(json.dumps(dic))


# read from txt file and transfer to json
def read(path):
    with open(path, 'r') as f:
        result = json.loads(f.read())
    return result


def save_list(lst, path):
    f = open(path, 'w')
    for i in lst:
        f.write((str)(i))
        f.write('\n')
    f.close()


def set_prefix(prefix, name):
    if not os.path.isdir(prefix):
        os.mkdir(prefix)
    if platform.system() == 'Windows':
        name = name.split('\\')[-1]
    else:
        name = name.split('/')[-1]
    shutil.copy(name, os.path.join(prefix, name))


def to_variable(x, has_gpu, requires_grad=False):
    if has_gpu:
        x = Variable(x.cuda(), requires_grad=requires_grad)
    else:
        x = Variable(x, requires_grad=requires_grad)
    return x


def get_parent_diectory(name, num):
    """
    return the parent directory
    :param name: __file__
    :param num: parent num
    :return: path
    """
    root = os.path.dirname(name)
    for i in range(num):
        root = os.path.dirname(root)
    return root


def single2tensor(path, transform):
    """
    convert a single image to tensor
    :param path: image path in disk
    :param transform: image preprocessing in form of transforms.Compose([transforms.ToTensor()])
    :return:
    """
    assert transform is not None, ''
    img_pil = Image.open(path)
    img_tensor = transform(img_pil)
    img_tensor.unsqueeze_(0)
    return img_tensor


def write_list(lst, path):
    if not isinstance(lst, list):
        raise TypeError('parameter lst must be list.')
    file = open(path, 'w')
    for var in lst:
        file.writelines(var)
        file.write('\n')
    file.close()


def txt2list(path):
    with open(path) as f:
        content = f.readlines()
    return [x.strip() for x in content]


def to_np(x):
    return x.data.cpu().numpy()


def add_prefix(prefix, path):
    return os.path.join(prefix, path)


def weight_to_cpu(path, is_load_on_cpu=True):
    if is_load_on_cpu:
        weights = torch.load(path, map_location=lambda storage, loc: storage)
        return remove_prefix(weights)
    else:
        return torch.load(path)


def remove_prefix(weights):
    new_state_dict = OrderedDict()
    for k, v in weights.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict


def merge_dict(dic1, dic2):
    merge = dic1.copy()
    merge.update(dic2)
    return merge


def to_image_type(x):
    x = torch.squeeze(x)
    x = to_np(x)

    x = np.transpose(x, (1, 2, 0))
    x = np.clip(x * 255, 0, 255).astype(np.uint8)
    return x


def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray


def get_mean_and_std(path,
                     transform=transforms.Compose([transforms.ToTensor()]),
                     channels=3):
    from utils.read_data import EasyDR
    dataset = EasyDR(path, None, transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
    mean = torch.zeros(channels)
    std = torch.zeros(channels)
    print('==> Computing mean and std..')
    for inputs, targets, _, _ in dataloader:
        for i in range(channels):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    mean, std = mean.numpy().tolist(), std.numpy().tolist()
    return [round(x, 4) for x in mean], [round(y, 4) for y in std]


def read_typical_images(path):
    with open(path, 'r') as f:
        dic = eval(f.read())
        return dic


def img2numpy(im):
    im2arr = np.array(im)
    return im2arr


def rgb2bgr(im):
    return im[:, :, ::-1]


def show_numpy(im, cmap):
    plt.imshow(im, cmap=cmap)
    plt.show()


# def append2xlsx(info, path):
#     """
#     @:param info:a list containing sublist which means a row in .xlsx file
#     reference: https://stackoverflow.com/questions/45103927/appending-rows-in-excel-xlswriter
#     """
#     wb = load_workbook(path)
#     # Select First Worksheet
#     ws = wb.worksheets[0]
#     for row in info:
#         ws.append(row)
#
#     wb.save(path)


def get_today():
    today = datetime.date.today()
    return str(today)

if __name__ == '__main__':
    # data_dir = '../data/diabetic_normal/train'
    # data_dir = '../data/mnist/train'
    # data_dir = '../data/xray_all/train'
    # data_dir = '../data/target/train'
    data_dir = '../data/split_contrast_dataset/train'
    print(get_mean_and_std(data_dir))