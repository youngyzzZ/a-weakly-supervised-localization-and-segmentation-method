import os
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.nn import DataParallel
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from networks.discriminator import get_discriminator
from networks.resnet import resnet18
from networks.unet import UNet
# from utils.Logger import Logger
from utils.read_data import ConcatDataset
from utils.util import add_prefix, weight_to_cpu, rgb2gray, write_list, copy, write
from tensorboardX import SummaryWriter

plt.switch_backend('agg')


class base(object):
    def __init__(self, args):
        """
        """
        self.debug = args.debug
        self.prefix = args.prefix
        self.pretrain_unet_path = args.pretrain_unet_path
        self.is_pretrained_unet = args.is_pretrained_unet

        self.use_gpu = torch.cuda.is_available()
        self.epoch_interval = 1 if self.debug else 5
        self.power = args.power
        self.data = args.data
        self.batch_size = args.batch_size
        self.epsi = args.epsi

        self.gan_type = args.gan_type
        self.u_depth = args.u_depth
        self.d_depth = args.d_depth
        self.dowmsampling = args.dowmsampling
        self.lr = args.lr
        self.beta1 = args.beta1
        self.eta = args.eta
        self.interval = args.interval
        self.epochs = args.epochs
        self.local = args.local
        # self.logger = Logger(add_prefix(self.prefix, 'tensorboard'))
        self.mean, self.std = 0.5, 0.5

        self.dataloader = self.get_dataloader()
        self.auto_encoder = self.get_unet()
        self.d = get_discriminator(self.gan_type, self.d_depth, self.dowmsampling)

        self.log_lst = []

        self.tb = SummaryWriter(log_dir=os.path.join(self.prefix, 'tb'))

        if self.use_gpu:
            self.auto_encoder = DataParallel(self.auto_encoder, device_ids=[0]).cuda()
            self.d = DataParallel(self.d, device_ids=[0]).cuda()
        else:
            raise RuntimeError('there is no gpu available.')

        self.save_init_paras()
        self.get_optimizer()
        self.save_hyperparameters(args)

    def save_hyperparameters(self, args):
        write(vars(args), add_prefix(self.prefix, 'para.txt'))
        print('save hyperparameters successfully.')

    def train(self, epoch):
        pass

    def validate(self, epoch):
        self.d.eval()
        self.auto_encoder.eval()

        real_data_score = []
        fake_data_score = []
        for i, (lesion_data, _, lesion_names, _, real_data, _, normal_names, _) in enumerate(self.dataloader):
            if i > 2:
                break
            if self.use_gpu:
                lesion_data, real_data = lesion_data.cuda(), real_data.cuda()
            phase = 'lesion_data'
            prefix_path = '%s/epoch_%d/%s' % (self.prefix, epoch, phase)
            lesion_output = self.d(self.auto_encoder(lesion_data)[0])
            fake_data_score += list(lesion_output.squeeze().cpu().data.numpy().flatten())

            for idx in range(self.batch_size):
                single_image = lesion_data[idx:(idx + 1), :, :, :]
                single_name = lesion_names[idx]
                self.save_single_image(prefix_path, single_name, single_image)
                if self.debug:
                    if idx > 10:
                        break

            phase = 'normal_data'
            prefix_path = '%s/epoch_%d/%s' % (self.prefix, epoch, phase)
            normal_output = self.d(real_data)
            real_data_score += list(normal_output.squeeze().cpu().data.numpy().flatten())

            for idx in range(self.batch_size):
                single_image = real_data[idx:(idx + 1), :, :, :]
                single_name = normal_names[idx]
                self.save_single_image(prefix_path, single_name, single_image)
                if self.debug:
                    if idx > 10:
                        break

        prefix_path = '%s/epoch_%d' % (self.prefix, epoch)
        self.plot_hist('%s/score_distribution.png' % prefix_path, real_data_score, fake_data_score)
        # if not self.debug:
        #     torch.save(self.auto_encoder.state_dict(), add_prefix(prefix_path, 'g.pkl'))
        #     torch.save(self.d.state_dict(), add_prefix(prefix_path, 'd.pkl'))
        #     print('save model parameters successfully when epoch=%d' % epoch)
        torch.save(self.auto_encoder.state_dict(), add_prefix(prefix_path, 'g.pkl'))
        torch.save(self.d.state_dict(), add_prefix(prefix_path, 'd.pkl'))
        print('save model parameters successfully when epoch=%d' % epoch)

    def main(self):
        print('training start!')
        start_time = time.time()

        for epoch in range(1, self.epochs + 1):
            # self.u_lr_scheduler.step()
            # self.c_lr_scheduler.step()
            # self.d_lr_scheduler.step()

            self.train(epoch)

            self.u_lr_scheduler.step()
            self.d_lr_scheduler.step()
            if epoch % self.epoch_interval == 0:
                self.validate(epoch)
        self.validate(self.epochs)

        total_ptime = time.time() - start_time
        print('Training complete in {:.0f}m {:.0f}s'.format(
            total_ptime // 60, total_ptime % 60))

    def save_init_paras(self):
        if not os.path.exists(self.prefix):
            os.makedirs(self.prefix)

        torch.save(self.auto_encoder.state_dict(), add_prefix(self.prefix, 'init_g_para.pkl'))
        torch.save(self.d.state_dict(), add_prefix(self.prefix, 'init_d_para.pkl'))
        print('save initial model parameters successfully')

    def get_optimizer(self):
        self.u_optimizer = torch.optim.Adam(self.auto_encoder.parameters(), lr=self.lr, betas=(self.beta1, 0.9))
        self.d_optimizer = torch.optim.Adam(self.d.parameters(), lr=self.lr, betas=(self.beta1, 0.9))
        self.u_lr_scheduler = lr_scheduler.ExponentialLR(self.u_optimizer, gamma=self.epsi)
        self.d_lr_scheduler = lr_scheduler.ExponentialLR(self.d_optimizer, gamma=self.epsi)

    def restore(self, x):
        x = torch.squeeze(x)
        x = x.data.cpu()
        for t, m, s in zip([x], [self.mean], [self.std]):
            t.mul_(s).add_(m)
        # transform Tensor to numpy
        x = x.numpy()
        # x = np.transpose(x, (1, 2, 0))
        x = np.clip(x * 255, 0, 255).astype(np.uint8)
        return x

    def get_unet(self):
        unet = UNet(1, depth=self.u_depth, in_channels=1)
        print(unet)
        print('load uent with depth %d and downsampling will be performed for %d times!!' % (
            self.u_depth, self.u_depth - 1))
        if self.is_pretrained_unet:
            unet.load_state_dict(weight_to_cpu(self.pretrain_unet_path))
            print('load pretrained unet!')
        return unet

    def get_dataloader(self):
        if self.local:
            raise ValueError('')
        else:
            print('load data from data center.')
            if self.data is not None:
                print('loading train data')
            else:
                raise ValueError("the parameter data must be in ['./data/gan']")
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])
        dataset = ConcatDataset(data_dir=self.data,
                                transform=transform,
                                alpha=self.power
                                )
        data_loader = DataLoader(dataset,
                                 batch_size=self.batch_size,
                                 shuffle=True,
                                 num_workers=2,
                                 drop_last=True,
                                 pin_memory=True if self.use_gpu else False)
        return data_loader

    def save_single_image(self, saved_path, name, inputs):
        """
        save unet output as a form of image
        """
        if not os.path.exists(saved_path):
            os.makedirs(saved_path)
        output, gdp = self.auto_encoder(inputs)

        left = self.restore(inputs)
        right = self.restore(output)
        lesion = self.restore(gdp)

        diff = np.where(left > right, left - right, right - left).clip(0, 255).astype(np.uint8)
        plt.figure(num='unet result', figsize=(8, 8))

        plt.subplot(2, 3, 1)
        plt.title('source image')
        plt.imshow(left, cmap='gray')
        plt.axis('off')

        plt.subplot(2, 3, 2)
        plt.title('unet output')
        plt.imshow(right, cmap='gray')
        plt.axis('off')

        plt.subplot(2, 3, 3)
        plt.title('GDP')
        plt.imshow(lesion, cmap='gray')
        plt.axis('off')

        plt.subplot(2, 3, 4)
        plt.imshow(diff, cmap='jet')
        plt.colorbar(orientation='horizontal')
        plt.title('difference in heatmap')
        plt.axis('off')

        plt.subplot(2, 3, 5)
        plt.imshow(diff.clip(0, 32), cmap='jet')
        plt.colorbar(orientation='horizontal')
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(add_prefix(saved_path, name))
        plt.close()

    def plot_hist(self, path, real_data, fake_data):
        bins = np.linspace(min(min(real_data), min(fake_data)), max(max(real_data), max(fake_data)), 60)
        plt.hist(real_data, bins=bins, alpha=0.3, label='real_score', edgecolor='k')
        plt.hist(fake_data, bins=bins, alpha=0.3, label='fake_score', edgecolor='k')
        plt.legend(loc='upper right')
        plt.savefig(path)
        plt.close()

    def save_log(self):
        write_list(self.log_lst, add_prefix(self.prefix, 'log.txt'))
        print('save running log successfully')

    def save_running_script(self, script_path):
        """
        save the main running script to get differences between scripts
        """
        copy(script_path, add_prefix(self.prefix, script_path.split('/')[-1]))

    def get_lr(self):
        lr = []
        for param_group in self.d_optimizer.param_groups:
            lr += [param_group['lr']]
        return lr[0]
