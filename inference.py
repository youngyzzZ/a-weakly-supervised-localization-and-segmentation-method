import argparse
import torch
import os
from torchvision import transforms
from utils.read_data import Covid_19_Dataset
from torch.utils.data import DataLoader
from networks.unet import UNet
from utils.util import weight_to_cpu
import numpy as np
from tqdm import tqdm
import cv2
from paraser import get_parser


def parse_args():
    parser = argparse.ArgumentParser(description='Segment Covid-19 lesion area')
    parser.add_argument('--test_set_dir', '-td', type=str, default='./data_1/covid_cell',
                        help='folder to load test set')
    parser.add_argument('--result_url', '-sd', type=str, default='./result/covid-cell_90',
                        help='folder to save result')
    parser.add_argument('--batch_size', '-bs', type=int, default=4, help='batch size')
    parser.add_argument('--power', '-k', type=int, default=2, help='power of weight')
    parser.add_argument('--input_channel', type=int, default=1, help='input channel of model')
    parser.add_argument('--u_depth', type=int, default=5, help='unet dpeth')
    parser.add_argument('--pretrain_unet_path', type=str, default='./check_points/train_11_ep_90/g.pkl',
                        help='pretrained unet')
    parser.add_argument('--threshold', type=float, default=60, help='segmentation threshold')
    return parser.parse_args()


class Predict:
    def __init__(self, args):
        self.mean, self.std = 0.5, 0.5
        self.args = args
        self.dataloader = self.get_dataloader()
        self.auto_encoder = self.get_unet()
        self.prediction = []
        self.label = []
        self.segmentation_threshold = args.threshold

    def get_dataloader(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])
        dataset = Covid_19_Dataset(data_dir=self.args.data_url,
                                   transform=transform)
        data_loader = DataLoader(dataset,
                                 batch_size=self.args.batch_size,
                                 shuffle=True,
                                 num_workers=2,
                                 drop_last=True,
                                 pin_memory=False)
        return data_loader

    def get_unet(self):
        unet = UNet(num_classes=1, depth=self.args.u_depth, in_channels=self.args.input_channel)
        print('loading the parameters of generator......')
        params_path = os.listdir(self.args.train_url)[0]
        unet.load_state_dict(weight_to_cpu(os.path.join(self.args.train_url, params_path)))
        # unet.load_state_dict(weight_to_cpu(self.args.pretrain_model))
        print('success loading!')
        return unet

    def save_single_image(self, saved_dir, names, inputs):
        """
        save unet output as a form of image
        """
        if not os.path.exists(saved_dir):
            os.makedirs(saved_dir)
        with torch.no_grad():
            outputs, gdps = self.auto_encoder(inputs)
        for i in range(inputs.shape[0]):
            input = inputs[i]
            output = outputs[i]
            name = names[i]
            left = self.restore(input)
            right = self.restore(output)

            diff = np.where(left > right, left - right, right - left).clip(0, 255).astype(np.uint8)
            diff_1 = np.where(diff >= self.segmentation_threshold, 255, 0).astype(np.uint8)
            heatmap = cv2.applyColorMap(np.uint8(diff), cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(saved_dir, 'hm_' + name), heatmap)
            cv2.imwrite(os.path.join(saved_dir, 'sg_' + name), diff_1)

    def restore(self, x):
        x = torch.squeeze(x)
        x = x.data.cpu()
        for t, m, s in zip([x], [self.mean], [self.std]):
            t.mul_(s).add_(m)
        # transform Tensor to numpy
        x = x.numpy()
        x = np.clip(x * 255, 0, 255).astype(np.uint8)
        return x

    def predictor(self):
        self.auto_encoder.eval()

        save_dir = self.args.result_url
        print(save_dir)

        for idx, item in tqdm(enumerate(self.dataloader), desc='processing result...', total=len(self.dataloader),
                              ncols=100):
            inputs, names = item['image'], item['name']
            self.save_single_image(save_dir, names, inputs)


if __name__ == '__main__':
    args = get_parser()
    predict = Predict(args)
    predict.predictor()
