"""
Read images and corresponding labels.
"""
import cv2
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
# from networks.unet import UNet
import torch

from torchvision.transforms import transforms

from utils.util import rgb2bgr, img2numpy


class EasyDR(Dataset):
    """
    a easy-classified diabetic retina dataset with clearly feature for normal and lesion images respectively
    """

    def __init__(self, data_dir, pre_transform, post_transform, alpha=6):
        """
        Args:
            data_dir: path to image directory.
            pre_transform: data augumentation such as RandomHorizontalFlip
            post_transform: image preprocessing such as Normalization and ToTensor
            alpha: (1+w)^\alpha using power function or (1+alpha*w) using linear function
        """
        assert alpha >= 0, 'the power parameter must be 0 at least!! '
        image_names = os.listdir(data_dir)
        self.data_dir = data_dir
        self.image_names = image_names
        self.pre_transform = pre_transform
        self.post_transform = post_transform
        self.class_names = ['lesion', 'normal']
        self.alpha = alpha

    def __getitem__(self, index):
        image_name = self.image_names[index]
        path = os.path.join(self.data_dir, image_name)
        image = Image.open(path)
        # convert RGB to GRAY
        if image.mode != 'L':
            image = image.convert('L')
        image = image.resize((64, 64))
        # lesion: 0 normal: 1
        if 'lesion' in image_name:
            label = 0
        elif 'normal' in image_name:
            label = 1
        else:
            raise ValueError('')
        if self.pre_transform is not None:
            image = self.pre_transform(image)
        gradient = (self._get_gradient_magnitude(image) + 1.0) ** self.alpha
        if self.post_transform is not None:
            image = self.post_transform(image)
        else:
            raise RuntimeError('')
        return image, label, image_name, torch.from_numpy(gradient)
        # return image, label, image_name

    @staticmethod
    def _get_gradient_magnitude(im):
        # convert r-g-b channels to b-g-r channels because cv2.imread loads image in b-g-r channels
        im = img2numpy(im)
        ddepth = cv2.CV_32F
        dx = cv2.Sobel(im, ddepth, 1, 0)
        dy = cv2.Sobel(im, ddepth, 0, 1)
        dxabs = cv2.convertScaleAbs(dx)
        dyabs = cv2.convertScaleAbs(dy)
        mag = cv2.addWeighted(dxabs, 0.5, dyabs, 0.5, 0)
        # mag = cv2.cvtColor(mag, cv2.COLOR_RGB2GRAY)
        mag = (mag / 255.0).astype('float32')
        # print(mag.min(), mag.max())
        # plt.imshow(mag, cmap='gray')
        # plt.show()
        return mag

    def __len__(self):
        return len(self.image_names)


class ConcatDataset(Dataset):
    def __init__(self, data_dir, transform, alpha=6):
        """
        train simultaneously on two datasets
        Args:
            data_dir: path to image directory.
            alpha: (1+w)^\alpha using power function
        reference: https://discuss.pytorch.org/t/train-simultaneously-on-two-datasets/649/16
        """
        assert alpha >= 0, 'the power parameter must be 0 at least!! '
        image_names = {x: os.listdir(os.path.join(data_dir, x)) for x in ['lesion', 'normal']}
        # assert len(image_names['lesion']) == len(image_names['normal']), ''
        self.data_dir = data_dir
        self.image_names = image_names
        self.transform = transform
        self.alpha = alpha

    def __getitem__(self, index):
        lesion_image, lesion_label, lesion_name, lesion_gradient = self.__get_data(index, 'lesion')
        normal_image, normal_label, normal_name, normal_gradient = self.__get_data(index, 'normal')

        return lesion_image, lesion_label, lesion_name, lesion_gradient, normal_image, normal_label, normal_name, normal_gradient

    def __get_data(self, index, phase):
        name = self.image_names[phase][index]
        path = '%s/%s/%s' % (self.data_dir, phase, name)
        image = Image.open(path)
        image = image.resize((256, 256))
        # convert RGB to GRAY
        if image.mode != 'L':
            image = image.convert('L')
        gradient = (self._get_gradient_magnitude(image) + 1.0) ** self.alpha
        if self.transform is not None:
            image = self.transform(image)
        if 'lesion' in path:
            label = 0
        elif 'normal' in path:
            label = 1
        else:
            raise ValueError('')
        return image, label, name, torch.from_numpy(gradient)

    @staticmethod
    def _get_gradient_magnitude(im):
        # convert r-g-b channels to b-g-r channels because cv2.imread loads image in b-g-r channels
        im = img2numpy(im)
        ddepth = cv2.CV_32F
        dx = cv2.Sobel(im, ddepth, 1, 0)
        dy = cv2.Sobel(im, ddepth, 0, 1)
        dxabs = cv2.convertScaleAbs(dx)
        dyabs = cv2.convertScaleAbs(dy)
        mag = cv2.addWeighted(dxabs, 0.5, dyabs, 0.5, 0)
        # mag = cv2.cvtColor(mag, cv2.COLOR_RGB2GRAY)
        mag = (mag / 255.0).astype('float32')
        # print(mag.min(), mag.max())
        # plt.imshow(mag, cmap='gray')
        # plt.show()
        return mag

    def __len__(self):
        return len(self.image_names['lesion'])


class Covid_19_Dataset(Dataset):
    def __init__(self, data_dir, transform):
        self.data_dir = data_dir
        self.transform = transform
        self.image_names = {x: os.listdir(os.path.join(data_dir, x)) for x in ['lesion', 'normal']}
        self.image_paths = self.__get_image_paths()
        # print(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image_name = os.path.basename(image_path)
        image = Image.open(image_path)
        image = image.resize((256, 256))
        if image.mode != 'L':
            image = image.convert('L')
        if self.transform is not None:
            image = self.transform(image)
        return {'image': image, 'name': image_name}

    def __get_image_paths(self):
        image_paths = []
        lesion_images = self.image_names['lesion']
        normal_images = self.image_names['normal']
        for image in lesion_images:
            image_paths.append(os.path.join(self.data_dir, 'lesion', image))
        for image in normal_images:
            image_paths.append(os.path.join(self.data_dir, 'normal', image))
        return image_paths

    def __len__(self):
        return len(self.image_paths)


def easy_dr_dataset():
    traindir = os.path.join('../data_1/encode_decode', 'train')
    valdir = os.path.join('../data_1/encode_decode', 'val')
    mean = 0.5
    std = 0.5
    normalize = transforms.Normalize(mean, std)
    pre_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(0.05, 0.05, 0.05, 0.05)
    ])
    post_transforms = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    train_dataset = EasyDR(traindir, pre_transforms, post_transforms, alpha=6)
    print(len(train_dataset))
    # print(train_dataset[0][0].shape)
    for item in train_dataset:
        print(item[0].shape, item[1], item[2], item[3].shape)


# #     # val_dataset = EasyDR(valdir, None, val_transforms, alpha=6)

def concat_dataset():
    transform = transforms.Compose([
        # transforms.Resize(124),
        transforms.ColorJitter(0.05, 0.05, 0.05, 0.05),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])
    dataset = ConcatDataset(data_dir='../../../dataset/weakly_supervised_seg',
                            transform=transform,
                            alpha=2
                            )
    for item in dataset:
        print(item[0].min(), item[0].max())
        print(item[0].shape, item[1], item[2], item[3].shape, item[4].shape, item[5], item[6], item[7].shape)
        break


if __name__ == '__main__':
    # easy_dr_dataset()
    concat_dataset()
