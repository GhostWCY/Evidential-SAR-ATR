from __future__ import print_function

import os
import time
from tqdm import tqdm
import torch
import numpy as np
import torch.nn.functional as F

from util import accuracy, AverageMeter, get_information


def write_file(filepath, string):
    with open(filepath, 'a') as af:
        af.write(string)
        af.write('\n')


def relu_evidence(y):
    return F.relu(y)


def model_test_complex(data_loader, model, device, opt, title, fw=True):
    logits_list = []
    prob_list = []
    prob_edl_list = []
    pred_label_list = []
    true_label_list = []
    uncertainty_list = []
    top1 = AverageMeter()
    if data_loader != None:
        if fw:
            path = opt.result_path
            filename = title + '.txt'
            file_path = os.path.join(path, filename)
            write_file(file_path, '#' * 15 + title)
        """One epoch validation"""
        batch_time = AverageMeter()
        # switch to evaluate mode
        model.eval()
        with torch.no_grad():
            with tqdm(data_loader, total=len(data_loader)) as pbar:
                end = time.time()
                for idx, (input, true_label, path) in enumerate(pbar):
                    if torch.cuda.is_available():
                        input = input.to(device)
                        true_label = true_label.to(device)

                    # compute output
                    logits = model(input)  # logits
                    topk = (1,)
                    maxk = max(topk)
                    _, pred_label = logits.topk(maxk, 1, True, True)
                    prob = F.softmax((logits), dim=1)

                    evidence = relu_evidence(logits)
                    alpha = evidence + 1
                    uncertainty = opt.n_way / torch.sum(alpha, dim=1, keepdim=True)
                    prob_edl = alpha / torch.sum(alpha, dim=1, keepdim=True)
                    # measure accuracy and record loss
                    acc1, acc2 = accuracy(logits, true_label, topk=(1, 2))
                    top1.update(acc1[0], input.size(0))
                    logits_list += (logits.tolist())
                    pred_label_list += (i for j in pred_label.tolist() for i in j)
                    true_label_list += (true_label.tolist())
                    prob_list += (prob.tolist())
                    prob_edl_list += (prob_edl.tolist())
                    uncertainty_list += (uncertainty.tolist())
                    # measure elapsed time
                    batch_time.update(time.time() - end)
                    end = time.time()

                    pbar.set_postfix({"Acc@1": '{0:.2f}'.format(top1.avg.cpu().numpy())})
                    if fw:
                        write_file(file_path, '*' * 10)
                        write_file(file_path, 'image_path:  ' + str(path))
                        write_file(file_path, 'logits:   ' + str(logits))
                        write_file(file_path, 'prob:   ' + str(prob))
                        write_file(file_path, 'prob_edl:   ' + str(prob_edl))
                        write_file(file_path, 'pred_label:   ' + str(pred_label))
                        write_file(file_path, 'true_label:   ' + str(true_label))
                        write_file(file_path, 'alpha:   ' + str(alpha))
                        write_file(file_path, 'uncertainty:   ' + str(uncertainty))
            print('Test_Acc@1 {top1.avg:.3f}'.format(top1=top1))
    return top1.avg, logits_list, prob_list, prob_edl_list, pred_label_list, true_label_list, uncertainty_list


def model_test(data_loader, model, device, opt, title, fw=True):
    top1 = AverageMeter()
    logits_list = []
    prob_list = []
    pred_label_list = []
    true_label_list = []
    if data_loader != None:
        if fw:
            result_path = opt.result_path
            filename = title + '.txt'
            file_path = os.path.join(result_path, filename)
            write_file(file_path, '#' * 15 + title)
        # switch to evaluate mode
        model.eval()
        with torch.no_grad():
            with tqdm(data_loader, total=len(data_loader)) as pbar:
                for idx, (input, true_label, path) in enumerate(pbar):
                    if torch.cuda.is_available():
                        input = input.to(device)
                        true_label = true_label.to(device)
                    # compute output
                    logits = model(input)  # logits
                    prob = F.softmax((logits), dim=1)
                    topk = (1,)
                    maxk = max(topk)
                    _, pred_label = logits.topk(maxk, 1, True, True)
                    # measure accuracy and record loss
                    acc1, acc2 = accuracy(logits, true_label, topk=(1, 2))
                    top1.update(acc1[0], input.size(0))
                    logits_list += (logits.tolist())
                    prob_list += (prob.tolist())
                    pred_label_list += (i for j in pred_label.tolist() for i in j)
                    true_label_list += (true_label.tolist())
                    pbar.set_postfix({"Acc@1": '{0:.2f}'.format(top1.avg.cpu().numpy())})
                    if fw:
                        write_file(file_path, '*' * 10)
                        write_file(file_path, 'image_path:  ' + str(path))
                        write_file(file_path, 'logits:   ' + str(logits))
                        write_file(file_path, 'pred_label:   ' + str(pred_label))
                        write_file(file_path, 'true_label:   ' + str(true_label))
            print('Test_Acc@1 {top1.avg:.3f}'.format(top1=top1))
    return top1.avg, logits_list, prob_list, pred_label_list, true_label_list

