import argparse
import sys
import torch
import os
from torchvision import transforms
from utils.read_data import Covid_19_Dataset
from torch.utils.data import DataLoader
from networks.unet import UNet
from collections import OrderedDict
from utils.util import weight_to_cpu
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import precision_recall_curve, auc
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Segment Covid-19 lesion area')
    parser.add_argument('--test_set_dir', '-td', type=str, default='./data_1/covid_19_ct',
                        help='folder to load test set')
    parser.add_argument('--save_dir', '-sd', type=str, default='./result/ep_62_2', help='folder to save result')
    parser.add_argument('--batch_size', '-bs', type=int, default=4, help='batch size')
    parser.add_argument('--power', '-k', type=int, default=2, help='power of weight')
    parser.add_argument('--input_channel', type=int, default=1, help='input channel of model')
    parser.add_argument('--u_depth', type=int, default=5, help='unet dpeth')
    parser.add_argument('--pretrain_unet_path', type=str, default='./check_points/train_3_ep_62/g.pkl',
                        help='pretrained unet')
    parser.add_argument('--gt_dir', '-gtd', type=str, default='./data_1/covid_19_ct/mask',
                        help='folder of ground truth')
    return parser.parse_args()


class Predict:
    def __init__(self, args):
        self.mean, self.std = 0.5, 0.5
        self.args = args
        self.dataloader = self.get_dataloader()
        self.auto_encoder = self.get_unet()
        self.prediction = []
        self.label = []
        self.segmentation_threshold = 60

    def get_dataloader(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])
        dataset = Covid_19_Dataset(data_dir=self.args.test_set_dir,
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
        unet.load_state_dict(weight_to_cpu(self.args.pretrain_unet_path))
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
        # print(outputs.shape, gdps.shape)
        total_dice = 0
        for i in range(inputs.shape[0]):
            input = inputs[i]
            output = outputs[i]
            gdp = gdps[i]
            name = names[i]
            left = self.restore(input)
            right = self.restore(output)
            lesion = self.restore(gdp)

            shotname, extension = os.path.splitext(name)

            # gt.resize muust use Nearest neighbor interpolation
            gt = Image.open(os.path.join(self.args.gt_dir, shotname + '_mask' + extension)).resize((256, 256),
                                                                                                   Image.NEAREST)
            gt = np.array(gt)
            diff = np.where(left > right, left - right, right - left).clip(0, 255).astype(np.uint8)
            diff_1 = np.where(diff >= self.segmentation_threshold, 255, 0).astype(np.uint8)
            self.save_prediction_label(diff, gt)
            dice_score = self.dice_coeff(diff_1 / 255, gt / 255)
            # print(dice_score)
            total_dice += dice_score
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
            plt.imshow(gt, cmap='gray')
            plt.title('GT')
            plt.axis('off')

            plt.subplot(2, 3, 6)
            plt.imshow(diff_1, cmap='gray')
            plt.title('segmentation result ' + str(round(dice_score, 3)))
            plt.axis('off')

            plt.tight_layout()
            plt.savefig(os.path.join(saved_dir, name))
            plt.close()
        return total_dice

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

    def dice_score(self, X, Y):
        assert X.shape == Y.shape
        return self.dice_coeff(X, Y)

    def dice_coeff(self, pred, target):
        smooth = 1.
        assert pred.shape == target.shape
        # num = pred.size(0)
        m1 = pred  # Flatten
        m2 = target  # Flatten
        intersection = (m1 * m2).sum()

        return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)

    def save_prediction_label(self, diff, gt):
        predict = (diff / 255.).reshape(-1)
        label = (gt / 255.).reshape(-1)
        self.prediction.extend(list(predict))
        self.label.extend(list(label))

    def show_PR_curve(self):
        print(len(self.label), len(self.prediction))
        precision, recall, thresholds = precision_recall_curve(self.label, self.prediction)
        pr_auc = auc(recall, precision)
        print('auc:', pr_auc)
        plt.plot(recall, precision, color='blue')
        plt.title('Precision/Recall Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend(loc="upper right", labels=['Ours(AUC={})'.format(round(pr_auc, 3))])
        plt.savefig(os.path.join(self.args.save_dir, 'PR_curve.png'))
        plt.show()

        np.savez(os.path.join(self.args.save_dir, 'PR_data.npz'), precision=precision, recall=recall)

    def predictor(self):
        self.auto_encoder.eval()

        save_dir = self.args.save_dir
        print(save_dir)
        total_num = 0
        total_dice = 0

        for idx, item in tqdm(enumerate(self.dataloader), desc='processing result...', total=len(self.dataloader),
                              ncols=100):
            inputs, names = item['image'], item['name']
            # print(inputs.shape, names)
            total_num += len(names)
            batch_dice = self.save_single_image(save_dir, names, inputs)
            total_dice += batch_dice
        print('average dice:', total_dice / total_num)


if __name__ == '__main__':
    args = parse_args()
    predict = Predict(args)
    predict.predictor()
    predict.show_PR_curve()
