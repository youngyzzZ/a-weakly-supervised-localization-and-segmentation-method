"""
usage:
python identical_mapping.py -b=64 -e=2 -p test -a=2 -i=20 -d=./data/easy_dr_128 -gpu=0,1,2,3 --debug
"""
import os
import sys
import argparse
import time

import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
from torch.backends import cudnn
from torch.nn import DataParallel
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

sys.path.append('./')
# from utils.Logger import Logger
from utils.read_data import EasyDR
from networks.unet import UNet
from utils.util import set_prefix, write, add_prefix, rgb2gray

plt.switch_backend('agg')

parser = argparse.ArgumentParser(description='Training on Covid-19 Dataset')
parser.add_argument('-b', '--batch_size', default=16, type=int,
                    help='batch size')
parser.add_argument('-e', '--epochs', default=1, type=int,
                    help='training epochs')
parser.add_argument('--cuda', default=torch.cuda.is_available(), type=bool,
                    help='use gpu or not')
parser.add_argument('-i', '--interval_freq', default=12, type=int,
                    help='printing log frequence')
parser.add_argument('-d', '--data', default='../../dataset/COVID-19-20_v2/reconstruct_1',
                    help='path to dataset')
parser.add_argument('-p', '--prefix', required=True, type=str,
                    help='folder prefix')
parser.add_argument('-a', '--alpha', default=6, type=int,
                    help='power of gradient weight matrix')
parser.add_argument('--unet_depth', type=int, default=5, help='unet depth')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--debug', action='store_true', default=False, help='in debug or not(default: false)')


def main():
    global args, logger
    args = parser.parse_args()
    # logger = Logger(add_prefix(args.prefix, 'logs'))
    set_prefix(args.prefix, __file__)
    model = UNet(1, depth=5, in_channels=1)
    print(model)
    print('load unet with depth=5')
    if args.cuda:
        model = DataParallel(model).cuda()
    else:
        raise RuntimeError('there is no gpu')
    criterion = nn.L1Loss(reduce=False).cuda()
    print('use l1_loss')
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # accelerate the speed of training
    cudnn.benchmark = True

    train_loader, val_loader = load_dataset()
    # class_names=['LESION', 'NORMAL']
    class_names = train_loader.dataset.class_names
    print(class_names)

    since = time.time()
    print('-' * 10)
    for epoch in range(args.epochs):
        train(train_loader, model, optimizer, criterion, epoch)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    validate(model, val_loader, train_loader)
    # save model parameter
    torch.save(model.state_dict(), add_prefix(args.prefix, 'identical_mapping.pkl'))
    # save running parameter setting to json
    write(vars(args), add_prefix(args.prefix, 'paras.txt'))


def load_dataset():
    global mean, std
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    mean = 0.5
    std = 0.5
    if args.data == '../../dataset/COVID-19-20_v2/reconstruct_1':
        print('load horizontal flipped DR with size 512 successfully!!')
    else:
        raise ValueError("parameter 'data' %s that means path to dataset must be in ['./data/target_128']" % args.data)
    normalize = transforms.Normalize(mean, std)
    pre_transforms = transforms.Compose([
        # transforms.Resize((124, 124)),
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
        # transforms.Resize((124, 124)),
        transforms.ToTensor(),
        normalize,
    ])
    train_dataset = EasyDR(traindir, pre_transforms, post_transforms, alpha=args.alpha)
    val_dataset = EasyDR(valdir, None, val_transforms, alpha=args.alpha)
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=4,
                              pin_memory=True if args.cuda else False)
    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=1,
                            pin_memory=True if args.cuda else False)
    return train_loader, val_loader


def restore(x):
    x = torch.squeeze(x)
    x = x.data.cpu()
    for t, m, s in zip([x], [mean], [std]):
        t.mul_(s).add_(m)
    # transform Tensor to numpy
    x = x.numpy()
    # x = np.transpose(x, (1, 2, 0))
    x = np.clip(x * 255, 0, 255).astype(np.uint8)

    return x


def train(train_loader, model, optimizer, criterion, epoch):
    model.train(True)
    print('Epoch {}/{}'.format(epoch + 1, args.epochs))
    # Iterate over data.
    for idx, (inputs, _, _, weights) in enumerate(train_loader):
        if args.cuda:
            inputs, weights = inputs.cuda(), weights.unsqueeze(1).cuda()
        optimizer.zero_grad()
        # forward
        outputs = model(inputs)[0]
        print(outputs[0].max())
        loss = (weights * criterion(outputs, inputs)).mean()
        loss.backward()

        optimizer.step()
        step = epoch * int(len(train_loader.dataset) / args.batch_size) + idx
        info = {'loss': loss.item()}
        # for tag, value in info.items():
        #     logger.scalar_summary(tag, value, step)
        if idx % args.interval_freq == 0:
            print('unet_loss: {:.4f}'.format(loss.item()))


def validate(model, val_loader, train_loader):
    class_names = val_loader.dataset.class_names
    for phase in ['train', 'val']:
        for name in class_names:
            saved_path = '%s/%s/%s' % (args.prefix, phase, name.lower())
            if not os.path.exists(saved_path):
                os.makedirs(saved_path)
    model.eval()

    # save a sample from validate dataset
    phase = 'train'
    sample_inputs, sample_labels, sample_images_name, _ = next(iter(train_loader))
    if args.cuda:
        sample_inputs = sample_inputs.cuda()
    batch_size = sample_inputs.size(0)
    for idx in range(batch_size):
        single_image = sample_inputs[idx:(idx + 1), :, :, :]
        single_label = sample_labels[idx: idx + 1]
        single_name = sample_images_name[idx]
        single_output = model(single_image)[0]
        save_single_image(single_name,
                          single_image,
                          single_output,
                          class_names[np.array(single_label)[0]].lower(),
                          phase)
        if args.debug:
            break
    phase = 'val'
    for idx, (inputs, labels, name, _) in enumerate(val_loader):
        if args.cuda:
            inputs = inputs.cuda()
        output = model(inputs)[0]
        # save single image
        save_single_image(name[0],
                          inputs,
                          output,
                          class_names[labels.numpy()[0]].lower(),
                          phase)
        if args.debug:
            break


def save_single_image(name, inputs, output, label, phase):
    """
    save single image local
    :param name: name
    :param inputs: network input
    :param output: network output
    :param label: image label: 'lesion' or 'normal'
    :param phase: image source: 'training' or 'validate' dataset
    :return:
    """
    left = restore(inputs)
    right = restore(output)
    plt.figure(num='unet result', figsize=(8, 8))
    plt.subplot(2, 2, 1)
    plt.title('source image')
    plt.imshow(left, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.title('output image')
    plt.imshow(right, cmap='gray')
    plt.axis('off')

    diff = np.where(left > right, left - right, right - left).clip(0, 255).astype(np.uint8)
    plt.subplot(2, 2, 3)
    # plt.imshow(rgb2gray(diff), cmap='jet')
    plt.imshow(diff, cmap='jet')
    plt.colorbar()

    plt.tight_layout()
    plt.savefig(add_prefix(args.prefix, '%s/%s/%s' % (phase, label, name)))
    plt.close()


if __name__ == '__main__':
    main()
