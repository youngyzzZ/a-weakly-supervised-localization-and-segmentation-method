import argparse
import torch


def get_parser():
    parser = argparse.ArgumentParser(description='Training Custom Defined Model')
    parser.add_argument('--data_url', type=str, default='../../dataset/weakly_supervised_seg', help='dataset dir')
    parser.add_argument('--train_url', '-p', type=str, default='./models',
                        help='parent folder to save output data')
    parser.add_argument('--result_url', '-sd', type=str, default='./result/covid-cell_90',
                        help='folder to save result')
    parser.add_argument('--training_strategies', '-ts', default='update_u_d',
                        choices=['update_c_d_u', 'add_normal_constraint', 'remove_lesion_constraint', 'update_u_d'],
                        help='training strategies')
    parser.add_argument('--is_pretrained_unet', '-u', action='store_true', help='pretrained unet or not')
    parser.add_argument('--pretrain_unet_path', type=str, default='./exp/identical_mapping_124/identical_mapping.pkl',
                        help='pretrained unet')
    parser.add_argument('--power', '-k', type=int, default=2, help='power of weight')
    parser.add_argument('--batch_size', '-b', default=1, type=int, help='batch size')
    parser.add_argument('--gan_type', type=str, default='multi_scale',
                        choices=['conv_bn_leaky_relu', 'multi_scale', 'top_multi_scale', 'middle_multi_scale'],
                        help='discriminator type')
    parser.add_argument('--u_depth', type=int, default=5, help='unet dpeth')
    parser.add_argument('--d_depth', type=int, default=7, help='discriminator depth')
    parser.add_argument('--dowmsampling', type=int, default=4, help='dowmsampling times in discriminator')
    parser.add_argument('--lr', default=2e-4, type=float, help='learning rate')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 in Adam')
    parser.add_argument('--interval', '-i', default=15, type=int, help='log print interval')
    parser.add_argument('--epochs', '-e', default=120, type=int, help='training epochs')
    parser.add_argument('--lmbda', '-l', type=float, help='weight of u between u and c')
    parser.add_argument('--sigma', '-s', type=float, help='weight of d loss')
    parser.add_argument('--gamma', '-g', type=float, help='weight of u in u & d')
    parser.add_argument('--alpha', '-a', type=float, help='weight of d in u & d')
    parser.add_argument('--theta', '-t', type=float, help='weight of total variation loss')
    parser.add_argument('--eta', type=float, default=10.0, help='gradient penalty')
    parser.add_argument('--epsi', type=float, default=1.0, help='learning rate exponential decay step')
    parser.add_argument('--pretrained_steps', type=int, default=0, help='pretrained steps')
    parser.add_argument('--debug', action='store_true', default=False, help='mode:training or debug')
    parser.add_argument('--gpu_counts', default=torch.cuda.device_count(), type=int, help='gpu nums')
    parser.add_argument('--local', action='store_true', default=False, help='data location')
    parser.add_argument('--input_channel', type=int, default=1, help='input channel of model')
    parser.add_argument('--pretrain_model', type=str, default='./check_points/train_11_ep_90/g.pkl',
                        help='pretrained unet')
    parser.add_argument('--threshold', type=float, default=60, help='segmentation threshold')
    return parser.parse_args()
