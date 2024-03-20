# 读取结果txt文档并进行处理
import re
from metric import uq_metrics, ECE
import os
import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from matplotlib import rcParams
import xlwt


def get_data(str_data, convert_type='float'):
    data_list_float = []
    temp1 = str_data.split('[[')[1]
    temp2 = temp1.split(']]')[0]
    temp3 = eval('[[' + temp2 + ']]')
    if convert_type == 'float':
        data_list_float = (np.array(temp3).astype(float).tolist())
    elif convert_type == 'int':
        data_list_float = (np.array(temp3).astype(int).tolist())
    return data_list_float


def get_data2(str_data, spilt_s=['[', ']']):
    data_list_float = []
    temp1 = str_data.split('tensor')[-1]
    temp2 = temp1.split(spilt_s[0])[1]
    temp3 = temp2.split(spilt_s[1])[0]
    temp4 = temp3.split(',')
    for i in range(len(temp4)):
        data_list_float.append(int(temp4[i]))
    return data_list_float


def find_str(str_data, start, end):
    start_id = str_data.find(start)
    end_id = str_data.find(end, start_id)
    string = str_data[start_id:end_id + len(end)]
    return string


# # 从txt文档中读取实验结果计算指标
# def prameter_results_id(file_path):
#     f = open(file_path)
#     lines = f.readlines()  # 直接将文件中按行读到list里，效果与方法2一样
#     image_path_list = []
#     logits_list = []
#     prob_list = []
#     prob_edl_list = []
#     pred_label_list = []
#     true_label_list = []
#     alpha_list = []
#     uncertainty_list = []
#     for line in lines:
#         if 'image_path:' in line:
#             new_line = re.split('( )', line)
#             image_path_list.append(new_line[-1])
#         elif 'logits:' in line:
#             logits_list.append(get_data(line))
#         elif 'prob:' in line:
#             prob_list.append(get_data(line))
#         elif 'prob_edl:' in line:
#             prob_edl_list.append(get_data(line))
#         elif 'pred_label:' in line:
#             # pred_label_list.append(get_data(line, convert_type = 'int'))
#             pred_label_list.append(get_data2(line, spilt_s=['[[', ']]'])[0])
#         elif 'true_label:' in line:
#             true_label_list.append(get_data2(line)[0])
#         elif 'alpha:' in line:
#             alpha_list.append(get_data(line))
#         elif 'uncertainty:' in line:
#             uncertainty_list.append(get_data(line))
#     f.close()  # 关
#     # print('prob_edl_list', prob_edl_list)
#     return true_label_list, pred_label_list, logits_list  # , prob_edl_list, prob_list


def draw_cdf(entropies_list, label_list):
    color_list = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'gray', 'pink']
    xaxs = np.linspace(0, np.log(10), 1000)
    plt.figure(figsize=(5, 5))
    for index, entropies in enumerate(entropies_list):
        cdf = np.histogram(entropies, bins=1000)[0];
        cdf = cdf / np.sum(cdf)
        cdf = np.cumsum(cdf)
        plt.plot(xaxs, cdf, label=label_list[index], color=color_list[index])
    plt.legend(loc='upper left')
    plt.xlim(0, np.log(10) + 0.03)
    plt.ylim(0, 1.01)
    plt.xlabel("Entropy")
    plt.ylabel("Probability")
    # plt.title("CDF for the entropy of the predictive distributions")
    plt.show()


# draw confusion_matrix
def draw_cmatrix(confusion_matrix, classes, filename):
    # init the matrix
    proportion = []
    length = len(confusion_matrix)
    for i in confusion_matrix:
        for j in i:
            temp = j / (np.sum(i))
            proportion.append(temp)
    pshow = []
    for i in proportion:
        pt = "%.2f%%" % (i * 100)
        pshow.append(pt)
    proportion = np.array(proportion).reshape(length, length)  # reshape(列的长度，行的长度)
    pshow = np.array(pshow).reshape(length, length)

    # confusion matrix style
    config = {
        "font.family": 'Times New Roman',  # 设置字体类型
    }
    rcParams.update(config)
    plt.imshow(proportion, interpolation='nearest', cmap=plt.cm.Blues)  # 按照像素显示出矩阵
    # (改变颜色：'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds','YlOrBr', 'YlOrRd',
    # 'OrRd', 'PuRd', 'RdPu', 'BuPu','GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn')
    # plt.title('confusion_matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=12)
    plt.yticks(tick_marks, classes, fontsize=12)

    # set the fig style
    # print(pshow)
    config = {
        "font.family": 'Times New Roman',  # 设置字体类型
    }
    rcParams.update(config)
    plt.imshow(proportion, interpolation='nearest', cmap=plt.cm.Blues)  # 按照像素显示出矩阵
    # (改变颜色：'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds','YlOrBr', 'YlOrRd',
    # 'OrRd', 'PuRd', 'RdPu', 'BuPu','GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn')
    # plt.title('confusion_matrix')
    # plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=12)
    plt.yticks(tick_marks, classes, fontsize=12)
    # caculate the accuracy and set the color bar
    iters = np.reshape([[[i, j] for j in range(length)] for i in range(length)], (confusion_matrix.size, 2))
    for i, j in iters:
        if (i == j):
            plt.text(j, i - 0.12, format(confusion_matrix[i, j]), va='center', ha='center', fontsize=10, color='white',
                     weight=5)  # 显示对应的数字
            plt.text(j, i + 0.12, pshow[i, j], va='center', ha='center', fontsize=10, color='white')
        else:
            plt.text(j, i - 0.12, format(confusion_matrix[i, j]), va='center', ha='center', fontsize=10)  # 显示对应的数字
            plt.text(j, i + 0.12, pshow[i, j], va='center', ha='center', fontsize=10)

    plt.ylabel('True label', fontsize=16)
    plt.xlabel('Predict label', fontsize=16)
    plt.tight_layout()
    # plt.savefig(filename)
    plt.show()


def CE_read(file_name, mode='id'):
    f = open(file_name)
    content = f.read()
    f.close()
    samples = content.split('#' * 15)[1:]
    # val_sample = samples[0::2]
    # test_sample = samples[1::2]
    save_data = xlwt.Workbook()
    root = file_name.replace('.txt','')
    name = 'result'
    sv = save_data.add_sheet(name)
    sv.write(0, 0, 'accurate')
    sv.write(0, 1, 'ece')
    sv.write(0, 2, 'nll')
    sv.write(0, 3, 'score_brier')
    sv.write(0, 4, 'score_entropy')
    for i in range(len(samples)):
        logits_list = []
        pred_label_list = []
        true_label_list = []
        # if type == 'val':
        #     data_batch = samples[i].split('*' * 10)[1:]
        # elif type == 'test':
        #     data_batch = samples[i].split('*' * 10)[1:]
        data_batch = samples[i].split('*' * 10)[1:]
        for data in data_batch:
            logit_str = find_str(data, 'logits', ']]')
            logit = get_data(logit_str)
            logits_list += logit

            pred_label_str = find_str(data, 'pred_label', ']]')
            pred_label = get_data(pred_label_str, 'int')
            pred_label_list += np.array(pred_label).flatten().tolist()

            true_label = find_str(data, 'true_label', ']')
            true_label = get_data2(true_label)
            true_label_list += true_label
        prob_list = F.softmax(torch.tensor(logits_list), dim=1).tolist()

        # true_label_list, pred_label_list, logits_list, prob_edl_list, prob_list = prameter_results_id(file_name)
        # true_label_list, pred_label_list, logits_list, prob_list = prameter_results_id(file_name)
        # ewcognition

        accurate = uq_metrics.accuracy(np.array(true_label_list), np.array(pred_label_list))
        # cmatrix = confusion_matrix(np.array(true_label_list), np.array(pred_label_list))
        # # draw_cmatrix(cmatrix, classes = ['T72', '2S1', 'BRDM2', 'ZSU234'], filename=file_name.split('.')[0]+'.png')
        # draw_cmatrix(cmatrix,
        #              classes=['2S1', 'BMP2(SN_9563)', 'BRDM_2', 'BTR70(SN_C71)', 'BTR_60', 'D7', 'T62', 'T72(SN_132)',
        #                       'ZIL131', 'ZSU_23_4'], filename=file_name.split('.')[0] + '.png')

        # uncertainty estimation

        # nll = -uq_metrics.ll(y_true=np.array(true_label_list), preds=np.array(prob_list))
        # print('nll of softmax:', nll, '\n')
        nll = -uq_metrics.ll(y_true=np.array(true_label_list), preds=np.array(prob_list))
        score_brier = uq_metrics.brier(y_true=np.array(true_label_list), preds=np.array(prob_list))
        score_entropy = uq_metrics.entropy(preds=np.array(prob_list)).mean()

        logits = torch.tensor(logits_list)
        labels = torch.tensor(true_label_list)
        ece_value = ECE._ECELoss()
        ece = ece_value(logits, labels)

        # print('accurate', accurate, '\n')
        # print('ece:', ece.item(), '\n')
        # print('nll:', nll, '\n')
        # print('score_brier:', score_brier, '\n')
        # print('score_entropy:', score_entropy, '\n')
        sv.write(i + 1, 0, accurate)
        sv.write(i + 1, 1, ece.item())
        sv.write(i + 1, 2, nll)
        sv.write(i + 1, 3, score_brier)
        sv.write(i + 1, 4, score_entropy)

        # preds_edl = np.array(prob_edl_list)
        # entropy_edl = uq_metrics.entropy(preds_edl)
        preds_sm = np.array(prob_list)
        entropy_sm = uq_metrics.entropy(preds_sm)
        # print(entropy_sm)
        # print('entropy_cacl', entropy_edl, entropy_sm)
        # return ece, entropy_edl, entropy_sm
    save_data.save(root + '.xlsx')
    # return ece, entropy_sm

def EDL_read(file_name, mode='id'):
    f = open(file_name)
    content = f.read()
    f.close()
    samples = content.split('#' * 15)[1:]
    save_data = xlwt.Workbook()
    root = file_name.replace('.txt','')
    name = 'result'
    sv = save_data.add_sheet(name)
    sv.write(0, 0, 'accurate')
    sv.write(0, 1, 'ece')
    sv.write(0, 2, 'nll_softmax')
    sv.write(0, 3, 'nll_edl')
    sv.write(0, 4, 'score_brier')
    sv.write(0, 5, 'score_entropy')
    sv.write(0, 6, 'uncertainty')
    for i in range(len(samples)):
        logits_list = []
        prob_list = []
        prob_edl_list = []
        pred_label_list = []
        true_label_list = []
        alpha_list = []
        uncertainty_list = []
        data_batch = samples[i].split('*' * 10)[1:]
        for data in data_batch:
            logit_str = find_str(data, 'logits', ']]')
            logit = get_data(logit_str)
            logits_list += logit

            prob_str = find_str(data, 'prob', ']]')
            prob = get_data(prob_str)
            prob_list += prob

            prob_edl_str = find_str(data, 'prob_edl', ']]')
            prob_edl = get_data(prob_edl_str)
            prob_edl_list += prob_edl

            pred_label_str = find_str(data, 'pred_label', ']]')
            pred_label = get_data(pred_label_str, 'int')
            pred_label_list += np.array(pred_label).flatten().tolist()

            true_label = find_str(data, 'true_label', ']')
            true_label = get_data2(true_label)
            true_label_list += true_label

            alpha_str = find_str(data, 'alpha', ']]')
            alpha = get_data(alpha_str)
            alpha_list += alpha

            uncertainty_str = find_str(data, 'uncertainty', ']]')
            uncertainty = get_data(uncertainty_str)
            uncertainty_list += uncertainty


        accurate = uq_metrics.accuracy(np.array(true_label_list), np.array(pred_label_list))
        # uncertainty estimation

        nll_softmax = -uq_metrics.ll(y_true=np.array(true_label_list), preds=np.array(prob_list))
        nll_edl = -uq_metrics.ll(y_true=np.array(true_label_list), preds=np.array(prob_edl_list))
        score_brier = uq_metrics.brier(y_true=np.array(true_label_list), preds=np.array(prob_edl_list))
        score_entropy = uq_metrics.entropy(preds=np.array(prob_edl_list)).mean()

        logits = torch.tensor(logits_list)
        labels = torch.tensor(true_label_list)
        ece_value = ECE._ECELoss()
        ece = ece_value(logits, labels)

        # print('accurate', accurate, '\n')
        # print('ece:', ece, '\n')
        # print('nll_softmax:', nll_softmax, '\n')
        # print('nll_edl:', nll_edl, '\n')
        # print('score_brier:', score_brier, '\n')
        # print('score_entropy:', score_entropy, '\n')

        preds_edl = np.array(prob_edl_list)
        entropy_edl = uq_metrics.entropy(preds_edl)
        preds_sm = np.array(prob_list)
        entropy_sm = uq_metrics.entropy(preds_sm)
        # print('entropy_cacl', entropy_edl, entropy_sm)
        uncertainty=np.array(uncertainty_list).mean()
        sv.write(i + 1, 0, accurate)
        sv.write(i + 1, 1, ece.item())
        sv.write(i + 1, 2, nll_softmax)
        sv.write(i + 1, 3, nll_edl)
        sv.write(i + 1, 4, score_brier)
        sv.write(i + 1, 5, score_entropy)
        sv.write(i + 1, 6, uncertainty)
    save_data.save(root + '.xlsx')
    # return ece, entropy_edl, entropy_sm

if __name__ == '__main__':
    file_path = r'.\data_results_new\EDL'
    file_list = os.listdir(file_path)
    print(file_list)
    for file in file_list:
        if '.txt' in file:
            # if 'iid' in file:
            print('*' * 10, file)
            if 'CE' in file_path:
                CE_read(os.path.join(file_path, file))
            elif 'EDL' in file_path:
                EDL_read(os.path.join(file_path, file))
    # fewshot 的结果为val和test相继出现,
    # data_uncertainty顺序 speckle 0.7 - 1.0 defocus -0.001 - -0.007 lowres 0.5 1
                # ece, entropy_sm = CE_read(os.path.join(file_path, file), type=type)

    # prior_ece, prior_entropy_edl, prior_entropy_sm = main('/home/user/zxy/zxy22/02FSDA/results/MSTAR/grass15-EDL_prior-EDL/T5/4ways-1shot_T5/prior-EDL_ood15_15grass_15.txt')
    # #ce
    # ece, entropy_edl, entropy_sm = main(r'/home/user/zxy/zxy22/02FSDA/results/MSTAR/grass15-EDL_prior-EDL/T5/4ways-1shot_T5/EDL_ood15_15grass_15.txt')
    # draw_cdf([entropy_sm,prior_entropy_sm, entropy_edl, prior_entropy_edl], ['softmax_edl','softmax_prior_edl', 'edl', ' prior-edl'])#画累计函数分布图像
    # draw_cdf([entropy_sm], ['softmax_edl'])
