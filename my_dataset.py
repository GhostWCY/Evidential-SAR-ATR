# https://blog.csdn.net/weixin_42263486/article/details/108302350?utm_medium=distribute.pc_relevant.none-task-blog-title-3&spm=1001.2101.3001.4242

import zipfile
import os
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import re
import random
from collections import defaultdict

# 解压数据到指定文件
def unzip(filename, dst_dir):
    z = zipfile.ZipFile(filename)
    z.extractall(dst_dir)


class MyDataset(Dataset):
    def __init__(self, transform=None, data_path=None):
        super(MyDataset, self).__init__()
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
        self.data = data
        self.label = label
        self.num_class = len(label_name)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, label, path



def get_data_MSTAR(setname, transform, num_workers, data_path, args):
    data_loader = MyDataset(transform=transform, data_path=data_path)
    if setname == 'train':
        batch_size = args.batch_size
        shuffle = True
    elif setname == 'test':
        batch_size = 1
        shuffle = False
    data = DataLoader(dataset=data_loader, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers)
    return data
