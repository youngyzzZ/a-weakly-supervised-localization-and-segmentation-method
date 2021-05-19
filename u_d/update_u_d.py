import torch
from torch import nn
from torch.autograd import grad
from torch.optim import lr_scheduler

from u_d.base import base
from utils.tv_loss import TVLoss


class update_u_d(base):
    def __init__(self, args):
        base.__init__(self, args)
        self.alpha = args.alpha
        self.sigma = args.sigma
        self.lmbda = args.lmbda
        self.gamma = args.gamma
        self.theta = args.theta
        self.pretrained_steps = args.pretrained_steps
        self.l1_criterion = nn.L1Loss(reduce=False).cuda()
        self.tv_loss_criterion = TVLoss().cuda()

    def train(self, epoch):
        l_n = len(self.dataloader)
        l_d_real_loss = 0.0
        l_d_fake_loss = 0.0
        l_d_loss = 0.0
        l_d_loss_ = 0.0
        l_u_loss_ = 0.0
        l_u_d_loss = 0.0

        self.d.train()
        self.auto_encoder.train()

        for idx, data in enumerate(self.dataloader, 1):
            step = (epoch - 1) * (len(self.dataloader.dataset) // self.batch_size) + idx
            lesion_data, lesion_labels, lesion_name, lesion_gradient, real_data, normal_labels, normal_name, normal_gradient = data
            if self.use_gpu:
                lesion_data, lesion_labels, normal_gradient = lesion_data.cuda(), lesion_labels.cuda(), normal_gradient.unsqueeze(
                    1).cuda()
                real_data, normal_labels, lesion_gradient = real_data.cuda(), normal_labels.cuda(), lesion_gradient.unsqueeze(
                    1).cuda()

            # training network: update d
            self.d_optimizer.zero_grad()
            # fake_data[0] is fake image without lesion, fake_data[1] is fake lesion
            fake_data, fake_lesion = self.auto_encoder(lesion_data)
            real_dis_output = self.d(real_data)
            fake_dis_output = self.d(fake_data.detach())

            theta = torch.rand((real_data.size(0), 1, 1, 1))
            if self.use_gpu:
                theta = theta.cuda()
            x_hat = theta * real_data.data + (1 - theta) * fake_data.data
            x_hat.requires_grad = True
            pred_hat = self.d(x_hat)
            if self.use_gpu:
                gradients = grad(outputs=pred_hat, inputs=x_hat, grad_outputs=torch.ones(pred_hat.size()).cuda(),
                                 create_graph=True, retain_graph=True, only_inputs=True)[0]
            else:
                gradients = grad(outputs=pred_hat, inputs=x_hat, grad_outputs=torch.ones(pred_hat.size()),
                                 create_graph=True, retain_graph=True, only_inputs=True)[0]
            gradient_penalty = self.eta * ((gradients.view(gradients.size()[0], -1).norm(2, 1) - 1) ** 2).mean()

            d_real_loss = -torch.mean(real_dis_output)
            d_fake_loss = torch.mean(fake_dis_output)

            # don't update D too much by set weight
            d_loss = self.sigma * (d_real_loss + d_fake_loss + gradient_penalty)
            d_loss.backward()
            self.d_optimizer.step()

            # track loss
            l_d_real_loss += d_real_loss.item()
            l_d_fake_loss += d_fake_loss.item()
            l_d_loss += d_loss.item()

            # update u
            if step > self.pretrained_steps:
                self.u_optimizer.zero_grad()
                dis_output = self.d(fake_data)
                d_loss_ = -torch.mean(dis_output)

                real_data_, real_lesion = self.auto_encoder(real_data)
                normal_l1_loss = (normal_gradient * self.l1_criterion((real_data_ + real_lesion), real_data)).mean()
                lesion_l1_loss = (lesion_gradient * self.l1_criterion((fake_data + fake_lesion), lesion_data)).mean()
                normal_l1_loss_ = (normal_gradient * self.l1_criterion(real_data_, real_data)).mean()
                u_loss_ = normal_l1_loss + normal_l1_loss_ + lesion_l1_loss
                u_d_loss = self.alpha * d_loss_ + self.gamma * u_loss_
                u_d_loss.backward()
                self.u_optimizer.step()

                # track loss
                l_d_loss_ += d_loss_.item()
                l_u_loss_ += u_loss_.item()
                l_u_d_loss += u_d_loss.item()

                w_distance = d_real_loss.item() + d_fake_loss.item()

                if idx % self.interval == 0:
                    log_1 = '[{}/{}] {}={}*({}(d_real_loss)+{}(d_fake_loss)+{}*{}(gradient_penalty)), '.format(epoch,
                                                                                                               self.epochs,
                                                                                                               d_loss,
                                                                                                               self.sigma,
                                                                                                               d_real_loss,
                                                                                                               d_fake_loss,
                                                                                                               self.eta,
                                                                                                               gradient_penalty)
                    log_2 = 'w_distance:{} ,'.format(w_distance)
                    log_3 = '{}={}({}(normal_l1_loss)+{}(normal_l1_loss_)+{}(lesion_l1_loss))+{}*{}(d_loss_)'.format(
                        u_d_loss,
                        self.gamma,
                        normal_l1_loss,
                        normal_l1_loss_,
                        lesion_l1_loss,
                        self.alpha, d_loss_)
                    log = log_1 + log_2 + log_3
                    print(log)
                    self.log_lst.append(log)

        self.tb.add_scalar('D_loss/d_real_loss', l_d_real_loss, epoch)
        self.tb.add_scalar('D_loss/d_fake_loss', l_d_fake_loss, epoch)
        self.tb.add_scalar('D_loss/d_loss', l_d_loss, epoch)
        self.tb.add_scalar('G_loss/d_loss_', l_d_loss_, epoch)
        self.tb.add_scalar('G_loss/u_loss_', l_u_loss_, epoch)
        self.tb.add_scalar('G_loss/u_d_loss', l_u_d_loss, epoch)
