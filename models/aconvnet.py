import torch.nn as nn
import torch
import collections

_activations = {
    'relu': nn.ReLU,
    'relu6': nn.ReLU6,
    'leaky_relu': nn.LeakyReLU
}

class BaseBlock(nn.Module):

    def __init__(self):
        super(BaseBlock, self).__init__()
        self._layer: nn.Sequential

    def forward(self, x):
        return self._layer(x)

class Conv2DBlock(BaseBlock):

    def __init__(self, shape, stride, padding=0, **kwargs):
        super(Conv2DBlock, self).__init__()

        h, w, in_channels, out_channels = shape
        _seq = collections.OrderedDict([
            ('conv', nn.Conv2d(in_channels, out_channels, kernel_size=(h, w), stride=stride, padding=padding))
        ])

        _bn = kwargs.get('batch_norm')
        if _bn:
            _seq.update({'bn': nn.BatchNorm2d(out_channels)})

        _act_name = kwargs.get('activation')
        if _act_name:
            _seq.update({_act_name: _activations[_act_name](inplace=True)})

        _max_pool = kwargs.get('max_pool')
        if _max_pool:
            _kernel_size = kwargs.get('max_pool_size', 2)
            _stride = kwargs.get('max_pool_stride', _kernel_size)
            _seq.update({'max_pool': nn.MaxPool2d(kernel_size=_kernel_size, stride=_stride)})

        self._layer = nn.Sequential(_seq)

        w_init = kwargs.get('w_init', None)
        idx = list(dict(self._layer.named_children()).keys()).index('conv')
        if w_init:
            w_init(self._layer[idx].weight)
        b_init = kwargs.get('b_init', None)
        if b_init:
            b_init(self._layer[idx].bias)
            
class AConvNet(nn.Module):

    def __init__(self, **kwargs):
        super(AConvNet, self).__init__()
        self.dropout_rate = kwargs.get('dropout_rate', 0.5) # 设置了dropout
        self.classes = kwargs.get('num_classes', 10)
        self.channels = kwargs.get('channels', 3)

        _w_init = kwargs.get('w_init', lambda x: nn.init.kaiming_normal_(x, nonlinearity='relu'))
        _b_init = kwargs.get('b_init', lambda x: nn.init.constant_(x, 0.1))

        self._layer = nn.Sequential(
            Conv2DBlock(
                shape=[5, 5, self.channels, 16], stride=1, padding='valid', activation='relu', max_pool=True,
                w_init=_w_init, b_init=_b_init
            ),
            Conv2DBlock(
                shape=[5, 5, 16, 32], stride=1, padding='valid', activation='relu', max_pool=True,
                w_init=_w_init, b_init=_b_init
            ),
            Conv2DBlock(
                shape=[6, 6, 32, 64], stride=1, padding='valid', activation='relu', max_pool=True,
                w_init=_w_init, b_init=_b_init
            ),
            Conv2DBlock(
                shape=[5, 5, 64, 128], stride=1, padding='valid', activation='relu',
                w_init=_w_init, b_init=_b_init
            ),
            nn.Dropout(p=self.dropout_rate),
            Conv2DBlock(
                shape=[3, 3, 128, self.classes], stride=1, padding=0,
                w_init=_w_init, b_init=nn.init.zeros_
            ),
            nn.AdaptiveAvgPool2d((1, 1)), # 如果不加则输入图像大小为88×88
            nn.Flatten()
        )

    def forward(self, x):
        return self._layer(x)

def aconvnet(**kwargs):
    return AConvNet(**kwargs)
