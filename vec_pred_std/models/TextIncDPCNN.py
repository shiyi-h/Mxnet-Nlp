from mxnet.gluon import nn
from mxnet import nd
from .Config import Config as BaseConfig

class Config(BaseConfig):
    """配置参数"""
    def __init__(self,*args, **kwargs):
        super(Config, self).__init__(*args, **kwargs)
        """ TextIncCNN层参数 """
        self.sing_sizes = [1, 1, 3, 3]
        self.dou_sizes = [(1, 3), (3, 5), (3, 3), (5, 5)]
        self.text_channels = 256
        """ 特征提取层参数 """
        self.kernel_size = 3
        self.num_channels = 250
        self.padding = 1
        self.drop_rate = 0.5
        """ ShortCut层参数"""
        self.shortcut_ksize = 3
        self.shortcut_channels = 250
        self.shortcut_pad = 1

class ShortCut(nn.Block):
    def __init__(self, channel_size:int, kernel_size:int, padding):
        super(ShortCut, self).__init__()
        self.maxpool = nn.MaxPool1D(pool_size=3, strides=2, prefix="ShortCut_Pool")
        self.conv = nn.Sequential(prefix="ShortCut_Conv")
        self.conv.add(
            nn.BatchNorm(),
            nn.Activation("relu"),
            nn.Conv1D(in_channels=channel_size,channels=channel_size, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm(),
            nn.Activation("relu"),
            nn.Conv1D(in_channels=channel_size,channels=channel_size, kernel_size=kernel_size, padding=padding))

    def forward(self, x):
        x = nd.concat(x, nd.zeros(shape=(x.shape[0],x.shape[1],1),ctx=x.context),dim=-1)
        x_shortcut = self.maxpool(x)
        x = self.conv(x_shortcut)
        return x + x_shortcut


class Model(nn.Block):
    def __init__(self, config,vocab, **kwargs):
        super(Model, self).__init__(**kwargs)
        self.constant_embedding = nn.Embedding(len(vocab), config.embed_size)
        self.embedding = nn.Embedding(len(vocab), config.embed_size)

        """ 与DPCNN不同点: region_embedding部分的卷积换成NiNCNN结构"""
        self.NiNConvs = nn.Sequential(prefix="NiNConvs_Layer")
        for kernel_size in config.sing_sizes:
            self.NiNConvs.add(self.NiNConv(config.embed_size*2,config.text_channels,kernel_size,double_convs=False))
        for kernel_size in config.dou_sizes:
            self.NiNConvs.add(self.NiNConv(config.embed_size*2,config.text_channels,kernel_size))
        self.change_dim_conv = nn.Conv1D(in_channels=config.text_channels, channels=config.num_channels,
                                         kernel_size=1, strides=1)

        """ NiN之后接个ShortCut_Conv """
        self.conv_block = nn.Sequential(prefix="After_NiN_Layer")
        self.conv_block.add(
            nn.BatchNorm(),
            nn.Activation("relu"),
            nn.Conv1D(in_channels=config.text_channels, channels=config.num_channels,
                      kernel_size=config.kernel_size, padding=config.padding),
            nn.BatchNorm(),
            nn.Activation("relu"),
            nn.Conv1D(in_channels=config.num_channels, channels=config.num_channels,
                      kernel_size=config.kernel_size, padding=config.padding)
        )

        """ 定义ShortCut层 """
        self.num_seq = len(config.sing_sizes)+len(config.dou_sizes) ##这里的num_seq应为NiN的卷积数
        shortcut_block_list = []
        while (self.num_seq>2):
            shortcut_block_list.append(ShortCut(config.shortcut_channels, config.shortcut_ksize, config.shortcut_pad))
            self.num_seq = self.num_seq // 2
        self.shortcut_layer = nn.Sequential(prefix="ShortCut_Layer")
        self.shortcut_layer.add(*shortcut_block_list)

        """ 定义输出层 """
        self.output = nn.Sequential(prefix="Output_Layer")
        self.output.add(
            nn.Dense(config.output_size),
            nn.BatchNorm(),
            nn.Activation("relu"),
            nn.Dropout(config.drop_rate),
            nn.Dense(config.output_size)
        )

    def NiNConv(self,in_channels,channels,kernel_size,double_convs=True):
        conv = nn.Sequential(prefix="NiNConv")
        if isinstance(kernel_size,(tuple,list)):
            k_size1,k_size2 = kernel_size[0],kernel_size[1]
        else:
            k_size1, k_size2 = kernel_size, kernel_size
        conv.add(
            nn.Conv1D(in_channels=in_channels, channels=channels, kernel_size=k_size1),
            nn.BatchNorm(),
            nn.Activation("relu"))
        if double_convs:
            conv.add(
                nn.Conv1D(in_channels=channels, channels=channels, kernel_size=k_size2),
                nn.BatchNorm(),
                nn.Activation("relu"),
                nn.GlobalMaxPool1D()
            )
        else:
            conv.add(nn.GlobalMaxPool1D())
        return conv

    def forward(self, x):
        x = nd.concat(self.embedding(x),self.constant_embedding(x), dim=2)
        x = x.transpose((0,2,1))
        x = [conv(x) for conv in self.NiNConvs]
        x = nd.concat(*x, dim=2)
        y = self.conv_block(x)
        x = self.change_dim_conv(x)
        x = x+y
        x = self.shortcut_layer(x)
        out = self.output(x)
        return out
