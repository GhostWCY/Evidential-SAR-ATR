# https://blog.csdn.net/weixin_42263486/article/details/108302350?utm_medium=distribute.pc_relevant.none-task-blog-title-3&spm=1001.2101.3001.4242

import zipfile
import os
import os.path as osp
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import random
import re
from collections import defaultdict


# 解压数据到指定文件
def unzip(filename, dst_dir):
    z = zipfile.ZipFile(filename)
    z.extractall(dst_dir)


# 新的采样函数，更简洁
class MyDataset_sample(Dataset):
    def __init__(self, random_seed, transform=None, data_path=None, args=None):
        super(MyDataset_sample, self).__init__()
        data_name = re.split('[/\\\]', data_path)[-2]
        if data_name == 'SOC':
            label_name = {'BMP2': 0, 'BTR70': 1, 'T72': 2, 'BTR_60': 3, '2S1': 4, 'BRDM_2': 5, 'D7': 6, 'T62': 7,
                          'ZIL131': 8, 'ZSU_23_4': 9}
        elif data_name == 'EOC-Depression':
            label_name = {'2S1': 0, 'BRDM_2': 1, 'ZSU_23_4': 2, 'T72': 3}
        elif data_name == 'EOC-Scene':
            label_name = {'BRDM_2': 0, 'ZSU_23_4': 1, 'T72': 2}
        elif data_name == 'EOC-Configuration-Version':
            label_name = {'T72': 0, 'BMP2': 1, 'BRDM_2': 2, 'BTR70': 3}

        self.transform = transform
        data = []
        label = []

        for root, dirs, files in os.walk(data_path):
            for file in files:
                if os.path.splitext(file)[1] == '.jpeg':
                    data.append(os.path.join(root, file))
                    for key, value in label_name.items():
                        if key in root:
                            label.append(value)
                            break

        self.num_class = len(label_name)
        self.sample = args.sample_flag
        if self.sample:
            label_to_paths = defaultdict(list)
            for path, label in zip(data, label):
                label_to_paths[label].append(path)
            # 设置随机种子，保证采样结果可复现
            random.seed(random_seed)
            # 从每个类别中随机抽取args.k_shot个样本
            sampled_paths_dict = {label: random.sample(paths, args.k_shot) if len(paths) >= args.k_shot else paths
                                  for label, paths in label_to_paths.items()}
            sample_data = []
            sample_label = []
            # 遍历sampled_paths字典，并填充path_list和label_list
            for label, paths in sampled_paths_dict.items():
                for path in paths:
                    sample_data.append(path)
                    sample_label.append(label)
            self.data = sample_data
            self.label = sample_label
        else:
            self.data = data
            self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, label, path


class MyDataset(Dataset):

    def __init__(self, setname, random_seed, transform=None, args=None):
        super(MyDataset, self).__init__()
        self.sample = args.sample_flag
        if setname == 'train':
            dirname = args.data_train_path
            label_list = os.listdir(dirname)
            print('label_list', label_list)
        elif setname == 'val':
            dirname = args.data_train_path
            label_list = os.listdir(dirname)
            print('label_list', label_list)
        elif setname == 'test':
            dirname = args.data_test_path
            label_list = os.listdir(dirname)
            print('label_list', label_list)
        else:
            raise ValueError('Unkown setname.')

        self.transform = transform
        self.setname = setname
        data = []
        label = []

        folders = [osp.join(dirname, label) for label in label_list if os.path.isdir(osp.join(dirname, label))]

        # Get the images' paths and labels
        for idx, this_folder in enumerate(folders):
            this_folder_images = os.listdir(this_folder)
            for image_path in this_folder_images:
                data.append(osp.join(this_folder, image_path))
                label.append(idx)

        self.data = data
        self.label = label
        self.num_class = len(set(label))

        # Sample from the dataset by n_ways, k-shots

        random.seed(random_seed)
        self.sample_label_train = []
        self.sample_label_test = []
        self.sample_data_train = []
        self.sample_data_test = []
        self.n_cls = args.n_way
        self.n_per = args.k_shot
        m_ind = []  # the data index of each class
        for i in range(max(self.label) + 1):
            ind = [k for k in range(len(self.label)) if self.label[k] == i]
            m_ind.append(ind)
        # random shuffle
        random.shuffle(m_ind)
        self.m_ind = m_ind

        # sample num_class indexs,e.g. 5

        classes = self.m_ind[:self.n_cls]
        for c in classes:
            random.shuffle(c)
            pos_train = c[:self.n_per]
            pos_val = c[self.n_per:]
            pos_test = c
            if setname == 'train':
                for m in pos_train:
                    self.sample_label_train.append(self.label[m])
                    self.sample_data_train.append(self.data[m])
            elif setname == 'val':
                for m in pos_val:
                    self.sample_label_test.append(self.label[m])
                    self.sample_data_test.append(self.data[m])
            elif setname == 'test':
                for m in pos_test:
                    self.sample_label_test.append(self.label[m])
                    self.sample_data_test.append(self.data[m])

    def __len__(self):
        if self.sample:
            if self.setname == 'train':
                return len(self.sample_data_train)
            elif self.setname == 'val':
                return len(self.sample_data_test)
            elif self.setname == 'test':
                return len(self.sample_data_test)
        else:
            return len(self.data)

    def __getitem__(self, i):
        if self.sample:
            if self.setname == 'train':
                path, label = self.sample_data_train[i], self.sample_label_train[i]
                image = self.transform(Image.open(path).convert('RGB'))
            elif self.setname == 'val':
                path, label = self.sample_data_test[i], self.sample_label_test[i]
                image = self.transform(Image.open(path).convert('RGB'))
            elif self.setname == 'test':
                path, label = self.sample_data_test[i], self.sample_label_test[i]
                image = self.transform(Image.open(path).convert('RGB'))
        else:
            path, label = self.data[i], self.label[i]
            image = self.transform(Image.open(path).convert('RGB'))
        return image, label, path


def get_data_MSTAR(setname, transform, num_workers, data_path, args, random_seed=None):
    # data_loader = MyDataset(setname, random_seed, transform=transform, args=args)
    if setname == 'train':
        batch_size = args.batch_size
        args.sample_flag = True
    elif setname == 'val':
        batch_size = args.batch_size
        args.sample_flag = True
    elif setname == 'test':
        batch_size = args.batch_size
        args.sample_flag = False
    data_loader = MyDataset_sample(random_seed, transform=transform, data_path=data_path, args=args)

    data = DataLoader(dataset=data_loader, batch_size=batch_size, shuffle=True,
                      num_workers=num_workers)
    return data
