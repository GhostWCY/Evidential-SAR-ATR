from __future__ import print_function

import os
import argparse
import time

from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn as nn

from util import accuracy, AverageMeter, get_information
import copy
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet18, resnet50, vgg16, vgg19, densenet121, mobilenet_v2, \
    inception_v3  # dmobilenet_v3
from models.aconvnet import aconvnet
from my_dataset import get_data_MSTAR
from model_eval import model_test, model_test_complex
from uncertainty_loss import edl_digamma_loss, edl_mse_loss, edl_log_loss
import random
import numpy as np
import re

current_path = os.path.dirname(__file__)
os.chdir(current_path)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--batch_size', type=int, default=8
                        , help='batch_size')
    parser.add_argument('--epochs', type=int, default=100, help='number of training epochs for MSTAR dataset')
    parser.add_argument('--sample_times', type=int, default=1,
                        help='number of sampling from MSTAR data to compose a few-shot dataset')
    parser.add_argument('--loss_option', type=str, default='CE',
                        choices=['CE', 'EDL'],
                        help='the choice of loss function, if CE in loss_option the parameter ''use_uncertainty'' must be false ')

    # uncertainty
    parser.add_argument('--use_uncertainty', type=bool, default=True, help='use uncertainty estimation')
    parser.add_argument('--uncertainty_loss', type=str, default='mse',
                        choices=['mse', 'log', 'digamma'],
                        help='use loss function for classification uncertainty estimation')

    # optimization
    parser.add_argument('--optim_option', type=str, default='Adam',
                        choices=['Adam', 'SGD'],
                        help='the choice of optimizer ')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='learning rate')  # 0.01
    parser.add_argument('--weight_decay', type=float, default=0.005, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.95, help='momentum, SGD only')

    # dataset and model
    parser.add_argument('--data_path', type=str, default='./MSTAR/SOC/TRAIN')  # 训练数据路径
    # parser.add_argument('--data_train_path', type=str, default='./data/MSTAR/SOC/TRAIN_17_DEG')  # 训练数据路径
    # parser.add_argument('--data_test_path', type=str, default='./data//MSTAR/SOC/TEST_15_DEG')  # 测试数据路径
    parser.add_argument('--result_root', type=str, default='./results/', help='txt path for writting test results')
    parser.add_argument('--result_path', type=str, default=None, help='txt path for writting test results')
    parser.add_argument('--save_modelpath', type=str, default='./model_trained/', help='path for saving models')
    parser.add_argument('--model_name', type=str, default='resnet18', help='model used in training')
    parser.add_argument('--n_way', type=int, default=10,
                        help='Number of classes for doing each classification run')

    # training parameters
    parser.add_argument('--seed', type=int, default=1, help='number of seed')
    parser.add_argument('--gpu', type=str, default='0', help='number of gpu')

    opt = parser.parse_args()
    return opt


def get_device(device_number):
    use_cuda = torch.cuda.is_available()
    print(use_cuda)
    if device_number == '1':
        device = torch.device("cuda:1" if use_cuda else "cpu")
        print(device)
    elif device_number == '0':
        device = torch.device("cuda:0" if use_cuda else "cpu")
        print(device)
    return device


def one_hot_embedding(labels, n_way=3):
    # Convert to One Hot Encoding
    y = torch.eye(n_way)
    return y[labels]


def y_embedding(y_hat, y):
    # Convert to One Hot Encoding
    y.append(y_hat)
    return y


def relu_evidence(y):
    return F.relu(y)


# image processing
image_size = 90
train_transform = transforms.Compose([
    transforms.Grayscale(3),
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

test_transform = transforms.Compose([
    transforms.Grayscale(3),
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


def train(train_loader, model, criterion_cls, optimizer, opt, epoch, device):
    """One epoch training"""
    model.train()
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    with tqdm(train_loader, total=len(train_loader)) as pbar:
        for i, (input, label, path) in enumerate(pbar):
            data_train = input.to(device)
            logit_s = model(data_train)  # prediction using deep learning model

            if opt.use_uncertainty:  # 使用不确定性估计
                y = one_hot_embedding(label, opt.n_way)  # one_hot 向量
                y = y.to(device)
                label = label.to(device)
                loss = criterion_cls(logit_s, y.float(), epoch, opt.n_way, 5, device)
            else:
                label = label.to(device)
                loss = criterion_cls(logit_s, label)

            acc1, acc2 = accuracy(logit_s, label, topk=(1, 2))
            top1.update(acc1[0], input.size(0))
            losses.update(loss.item(), input.size(0))
            # ===================backward=====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # ===================meters=====================
            batch_time.update(time.time() - end)
            end = time.time()

            pbar.set_postfix({"Acc@1": '{0:.2f}'.format(top1.avg.cpu().numpy()),
                              "Loss": '{0:.2f}'.format(losses.avg, 2),
                              })

    print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))

    return top1.avg, losses.avg


def model_selected(model_name):
    if model_name == 'resnet18':
        model = resnet18(pretrained=False, num_classes=opt.n_way)
    elif model_name == 'resnet50':
        model = resnet50(pretrained=False, num_classes=opt.n_way)
    elif model_name == 'vgg16':
        # model = vgg16(pretrained=False, num_classes=opt.n_way)
        model = vgg16(pretrained=True)
        print('model', model)
        model.classifier._modules['6'] = nn.Linear(4096, opt.n_way)

    elif model_name == 'vgg19':
        model = vgg19(pretrained=False, num_classes=opt.n_way)
    elif model_name == 'densenet121':
        model = densenet121(pretrained=False, num_classes=opt.n_way)
    elif model_name == 'mobilenet_v2':
        model = mobilenet_v2(pretrained=False, num_classes=opt.n_way)
    elif model_name == 'inception_v3':
        model = inception_v3(pretrained=False, aux_logits=False, num_classes=opt.n_way)
    elif model_name == 'aconvnet':
        model = aconvnet(num_classes=opt.n_way)
    # print('#' * 10, "model", model)
    return model


def get_test_loader(data_name):
    test_loader1 = None
    test_loader2 = None
    test_loader3 = None
    if data_name == 'SOC':
        SOC_test = get_data_MSTAR(setname='test', transform=test_transform, num_workers=0,
                                  data_path='./MSTAR/SOC/TEST',
                                  args=opt)  # get test data
        test_loader1 = SOC_test
    elif data_name == 'EOC-Depression':
        EOC_Depression_TEST_30 = get_data_MSTAR(setname='test', transform=test_transform, num_workers=0,
                                                data_path='./MSTAR/EOC-Depression/TEST_30',
                                                args=opt)  # get test data
        test_loader1 = EOC_Depression_TEST_30
    elif data_name == 'EOC-Scene':
        EOC_Scene_TEST = get_data_MSTAR(setname='test', transform=test_transform, num_workers=0,
                                        data_path='./MSTAR/EOC-Scene/TEST',
                                        args=opt)  # get test data
        EOC_Scene_TEST_30 = get_data_MSTAR(setname='test', transform=test_transform, num_workers=0,
                                           data_path='./MSTAR/EOC-Scene/TEST_30',
                                           args=opt)  # get test data
        EOC_Scene_TEST_45 = get_data_MSTAR(setname='test', transform=test_transform, num_workers=0,
                                           data_path='./MSTAR/EOC-Scene/TEST_45',
                                           args=opt)  # get test data
        test_loader1 = EOC_Scene_TEST
        test_loader2 = EOC_Scene_TEST_30
        test_loader3 = EOC_Scene_TEST_45
    elif data_name == 'EOC-Configuration-Version':
        EOC_Version = get_data_MSTAR(setname='test', transform=test_transform, num_workers=0,
                                     data_path='./MSTAR/EOC-Configuration-Version/TEST_Version',
                                     args=opt)  # get test data
        EOC_Configuration = get_data_MSTAR(setname='test', transform=test_transform, num_workers=0,
                                           data_path='./MSTAR/EOC-Configuration-Version/TEST_Configuration',
                                           args=opt)  # get test data
        test_loader1 = EOC_Version
        test_loader2 = EOC_Configuration

    return test_loader1, test_loader2, test_loader3


def main(opt):
    torch.manual_seed(opt.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  # opt.gpu
    device = get_device(opt.gpu)
    print('device', device)

    # loss function
    if opt.use_uncertainty:
        print('The loss contains EDL loss')
        if opt.uncertainty_loss == 'digamma':
            criterion_cls = edl_digamma_loss
        elif opt.uncertainty_loss == 'log':
            criterion_cls = edl_log_loss
        elif opt.uncertainty_loss == 'mse':
            criterion_cls = edl_mse_loss
        else:
            opt.error("--uncertainty requires --mse, --log or --digamma.")
    else:
        criterion_cls = nn.CrossEntropyLoss()
        print('the loss contains cross Entropy')

    data_name = re.split('[/\\\]', opt.data_path)[-2]
    for i in range(opt.sample_times):
        train_loader = get_data_MSTAR(setname='train', transform=train_transform, num_workers=0,
                                      data_path=opt.data_path,
                                      args=opt)  # get train data

        test_loader1, test_loader2, test_loader3 = get_test_loader(data_name)

        setup_seed(50)
        # model
        model = model_selected(opt.model_name)

        if opt.optim_option == 'Adam':
            optimizer = optim.Adam(model.parameters(),
                                   lr=opt.learning_rate,
                                   weight_decay=opt.weight_decay
                                   )
        elif opt.optim_option == 'SGD':
            optimizer = optim.SGD(model.parameters(),
                                  lr=opt.learning_rate,
                                  momentum=opt.momentum,
                                  weight_decay=opt.weight_decay)
        model.to(device)
        best_acc_test1 = 0
        best_acc_test2 = 0
        best_acc_test3 = 0
        best_model1 = None
        best_model2 = None
        best_model3 = None

        for epoch in range(1, opt.epochs + 1):
            print("==> training...")
            time1 = time.time()
            train_acc, train_loss = train(train_loader, model, criterion_cls, optimizer, opt, epoch, device)
            time2 = time.time()
            print(
                'epoch {}, total time {:.2f}, train accuracy {}, train loss {}'.format(epoch, time2 - time1, train_acc,
                                                                                       train_loss))
            if epoch % 3 == 0:
                if opt.use_uncertainty:
                    test_acc1, logits_list, prob_list, prob_edl_list, pred_label_list, true_label_list, uncertainty_list = model_test_complex(
                        test_loader1, model, device, opt, title='nll', fw=False)
                    test_acc2, logits_list, prob_list, prob_edl_list, pred_label_list, true_label_list, uncertainty_list = model_test_complex(
                        test_loader2, model, device, opt, title='nll', fw=False)
                    test_acc3, logits_list, prob_list, prob_edl_list, pred_label_list, true_label_list, uncertainty_list = model_test_complex(
                        test_loader3, model, device, opt, title='nll', fw=False)
                    print('test_acc2', test_acc2)
                    print('test_acc3', test_acc3)
                else:
                    test_acc1, logits_list, prob_list, pred_label_list, true_label_list = model_test(test_loader1,
                                                                                                     model, device, opt,
                                                                                                     title='nll',
                                                                                                     fw=False)
                    test_acc2, logits_list, prob_list, pred_label_list, true_label_list = model_test(test_loader2,
                                                                                                     model, device, opt,
                                                                                                     title='nll',
                                                                                                     fw=False)
                    test_acc3, logits_list, prob_list, pred_label_list, true_label_list = model_test(test_loader3,
                                                                                                     model, device, opt,
                                                                                                     title='nll',
                                                                                                     fw=False)
                if best_acc_test1 < test_acc1 and epoch > 80:
                    best_acc_test1 = test_acc1
                    best_model1 = copy.deepcopy(model)
                if best_acc_test2 < test_acc2 and epoch > 80:
                    best_acc_test2 = test_acc2
                    best_model2 = copy.deepcopy(model)
                if best_acc_test3 < test_acc3 and epoch > 80:
                    best_acc_test3 = test_acc3
                    best_model3 = copy.deepcopy(model)

        # save model
        if best_model1 != None:
            torch.save(best_model1, os.path.join(opt.save_modelpath,
                                                 opt.model_name + '_' + opt.loss_option + '_' + get_information(
                                                     opt.data_path) + str(best_acc_test1.cpu()) + '_test1.pth'))
        if best_model2 != None:
            torch.save(best_model2, os.path.join(opt.save_modelpath,
                                                 opt.model_name + '_' + opt.loss_option + '_' + get_information(
                                                     opt.data_path) + str(best_acc_test2.cpu()) + '_test2.pth'))
        if best_model3 != None:
            torch.save(best_model3, os.path.join(opt.save_modelpath,
                                                 opt.model_name + '_' + opt.loss_option + '_' + get_information(
                                                     opt.data_path) + str(best_acc_test3.cpu()) + '_test3.pth'))


if __name__ == '__main__':
    # chose the most popular 2 backbones in sar atr
    # model_list = ['resnet18', 'aconvnet']
    # data_list = ['./MSTAR/SOC/TRAIN', './MSTAR/EOC-Scene/TRAIN',
    #               './MSTAR/EOC-Depression/TRAIN', './MSTAR/EOC-Configuration-Version/TRAIN']
    # n_ways = [10, 3, 4, 4]
    model_list = ['resnet18']
    data_list = ['./MSTAR/EOC-Configuration-Version/TRAIN']
    n_ways = [4]

    opt = parse_option()
    for model_name in model_list:
        opt.model_name = model_name
        opt.result_path = os.path.join(opt.result_root, opt.loss_option + '/')
        if not os.path.exists(opt.result_path):
            os.makedirs(opt.result_path)
        for i in range(len(data_list)):
            if opt.loss_option == 'CE':
                opt.use_uncertainty = False
            elif opt.loss_option == 'EDL':
                opt.use_uncertainty = True
            opt.data_path = data_list[i]
            opt.n_way = n_ways[i]
            print(vars(opt))
            main(opt)
