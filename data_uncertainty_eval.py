from __future__ import print_function

import os
import argparse

import torch

import copy
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet18, resnet50, vgg16, vgg19, densenet121, mobilenet_v2, \
    inception_v3  # dmobilenet_v3

from models.aconvnet import aconvnet
from my_dataset import get_data_MSTAR
from model_eval import model_test, model_test_complex
import random
import numpy as np

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

    # dataset and model
    parser.add_argument('--result_root', type=str, default='./data_results/', help='txt path for writting test results')
    parser.add_argument('--result_path', type=str, default=None, help='txt path for writting test results')
    parser.add_argument('--save_modelpath', type=str, default='./model_trained/', help='path for saving models')
    parser.add_argument('--n_way', type=int, default=10,
                        help='Number of classes for doing each classification run')

    # training parameters
    parser.add_argument('--seed', type=int, default=1, help='number of seed')
    parser.add_argument('--gpu', type=str, default='0', help='number of gpu')

    opt = parser.parse_args()

    # set the path according to the environment

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


def get_test_loader(data_name, model_name, path):
    test_loader1 = None
    test_loader2 = None
    test_loader3 = None
    if data_name == 'SOC':
        SOC_test = get_data_MSTAR(setname='test', transform=test_transform, num_workers=0,
                                  data_path=path + '/TEST',
                                  args=opt)  # get test data
        test_loader1 = SOC_test
    elif data_name == 'EOC-Depression':
        EOC_Depression_TEST_30 = get_data_MSTAR(setname='test', transform=test_transform, num_workers=0,
                                                data_path=path + '/TEST_30',
                                                args=opt)  # get test data
        test_loader1 = EOC_Depression_TEST_30
    elif data_name == 'EOC-Scene':
        EOC_Scene_TEST = get_data_MSTAR(setname='test', transform=test_transform, num_workers=0,
                                        data_path=path + '/TEST',
                                        args=opt)  # get test data
        EOC_Scene_TEST_30 = get_data_MSTAR(setname='test', transform=test_transform, num_workers=0,
                                           data_path=path + '/TEST_30',
                                           args=opt)  # get test data
        EOC_Scene_TEST_45 = get_data_MSTAR(setname='test', transform=test_transform, num_workers=0,
                                           data_path=path + '/TEST_45',
                                           args=opt)  # get test data
        test_loader1 = EOC_Scene_TEST
        test_loader2 = EOC_Scene_TEST_30
        test_loader3 = EOC_Scene_TEST_45
    elif data_name == 'EOC-Configuration-Version':
        EOC_Version = get_data_MSTAR(setname='test', transform=test_transform, num_workers=0,
                                     data_path=path + '/TEST_Version',
                                     args=opt)  # get test data
        EOC_Configuration = get_data_MSTAR(setname='test', transform=test_transform, num_workers=0,
                                           data_path=path + '/TEST_Configuration',
                                           args=opt)  # get test data
        test_loader1 = EOC_Version
        test_loader2 = EOC_Configuration

    if 'test1' in model_name:
        return test_loader1
    elif 'test2' in model_name:
        return test_loader2
    elif 'test3' in model_name:
        return test_loader3
    # return test_loader1, test_loader2, test_loader3


def model_selected(model_name):
    if model_name == 'resnet18':
        model = resnet18(pretrained=False, num_classes=opt.n_way)
    elif model_name == 'resnet50':
        model = resnet50(pretrained=False, num_classes=opt.n_way)
    elif model_name == 'vgg16':
        model = vgg16(pretrained=False, num_classes=opt.n_way)
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


def main(opt):
    torch.manual_seed(opt.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  # opt.gpu
    # device = get_device(opt.gpu)

    conditions = ['SOC', 'EOC-Scene', 'EOC-Depression', 'EOC-Configuration-Version']
    n_ways = [10, 3, 4, 4]
    # test = ['TEST_15_DEG', 'TEST_30_DEG', 'TEST', 'TEST']
    root_list = ['speckle', 'defocus', 'lowres']
    speckles = ['1.0', '0.9', '0.8', '0.7']
    defocuses = ['-0.001', '-0.003', '-0.005', '-0.007']
    lowreses = ['0.5', '1']
    ori_path = './MSTAR/'
    lq_path = './data/'

    model_list = os.listdir(opt.save_modelpath)
    for model in model_list:
        if '_CE_' in model:
            opt.result_path = os.path.join(opt.result_root + 'CE/')
            if not os.path.exists(opt.result_path):
                os.makedirs(opt.result_path)

            for i in range(len(conditions)):
                if conditions[i] in model:
                    opt.n_way = n_ways[i]
                    net = torch.load(os.path.join(opt.save_modelpath, model))
                    # net = torch.load(os.path.join(opt.save_modelpath, model), map_location='cuda:0')
                    use_cuda = torch.cuda.is_available()
                    device = torch.device(next(net.parameters()).device if use_cuda else "cpu")

                    test_loader = get_test_loader(conditions[i], model, ori_path + conditions[i])
                    model_test(test_loader, copy.deepcopy(net), device, opt, title=model)

                    for root in root_list:
                        if 'speckle' in root:
                            for speckle in speckles:
                                path = lq_path + root + '/' + speckle + '/MSTAR/' + conditions[i]
                                test_loader = get_test_loader(conditions[i], model, path)
                                model_test(test_loader, copy.deepcopy(net), device, opt, title=model)
                        elif 'defocus' in root:
                            for defocus in defocuses:
                                path = lq_path + root + '/' + defocus + '/MSTAR/' + conditions[i]
                                test_loader = get_test_loader(conditions[i], model, path)
                                model_test(test_loader, copy.deepcopy(net), device, opt, title=model)
                        elif 'lowres' in root:
                            for lowres in lowreses:
                                path = lq_path + root + '/' + lowres + '/MSTAR/' + conditions[i]
                                test_loader = get_test_loader(conditions[i], model, path)
                                model_test(test_loader, copy.deepcopy(net), device, opt, title=model)
        elif '_EDL_' in model:
            opt.result_path = os.path.join(opt.result_root + 'EDL/')
            if not os.path.exists(opt.result_path):
                os.makedirs(opt.result_path)
            for i in range(len(conditions)):
                if conditions[i] in model:
                    opt.n_way = n_ways[i]
                    net = torch.load(os.path.join(opt.save_modelpath, model))
                    # net = torch.load(os.path.join(opt.save_modelpath, model), map_location='cuda:0')
                    use_cuda = torch.cuda.is_available()
                    device = torch.device(next(net.parameters()).device if use_cuda else "cpu")

                    test_loader = get_test_loader(conditions[i], model, ori_path + conditions[i])
                    model_test_complex(test_loader, copy.deepcopy(net), device, opt, title=model)

                    for root in root_list:
                        if 'speckle' in root:
                            for speckle in speckles:
                                path = lq_path + root + '/' + speckle + '/MSTAR/' + conditions[i]
                                test_loader = get_test_loader(conditions[i], model, path)
                                model_test_complex(test_loader, copy.deepcopy(net), device, opt, title=model)
                        elif 'defocus' in root:
                            for defocus in defocuses:
                                path = lq_path + root + '/' + defocus + '/MSTAR/' + conditions[i]
                                test_loader = get_test_loader(conditions[i], model, path)
                                model_test_complex(test_loader, copy.deepcopy(net), device, opt, title=model)
                        elif 'lowres' in root:
                            for lowres in lowreses:
                                path = lq_path + root + '/' + lowres + '/MSTAR/' + conditions[i]
                                test_loader = get_test_loader(conditions[i], model, path)
                                model_test_complex(test_loader, copy.deepcopy(net), device, opt, title=model)


if __name__ == '__main__':
    opt = parse_option()

    main(opt)
