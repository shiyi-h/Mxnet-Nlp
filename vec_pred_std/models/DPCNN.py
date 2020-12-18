from mxnet.gluon import nn
from mxnet import nd
from .Config import Config as BaseConfig



class Config(BaseConfig):
    """配置参数"""
    def __init__(self,*args, **kwargs):
        super(Config, self).__init__(*args, **kwargs)
        self.lr = 5e-4
        """ 特征提取层参数 """
        self.kernel_size = 3
        self.num_channels = 250
        self.padding = 1
        self.drop_rate = [0.2, 0.5] # 第一个为region_embedding层的dropout_rate, 第二个为输出层之前的dropput_rate
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
        x = x + x_shortcut
        return x


class Model(nn.Block):
    def __init__(self, config, vocab, **kwargs):
        super(Model, self).__init__(**kwargs)
        self.constant_embedding = nn.Embedding(len(vocab), config.embed_size)
        self.embedding = nn.Embedding(len(vocab), config.embed_size)

        """ 定义特征提取层卷积 """
        self.region_embedding = nn.Sequential(prefix="Region_Embedding_Layer")
        self.region_embedding.add(
            nn.Conv1D(in_channels=config.embed_size*2, channels=config.num_channels,
                      kernel_size=config.kernel_size, padding=config.padding),
            nn.BatchNorm(),
            nn.Activation("relu"),
            nn.Dropout(config.drop_rate[0])
        )
        self.conv_block = nn.Sequential(prefix="After_Region_Layer")
        self.conv_block.add(
            nn.BatchNorm(),
            nn.Activation("relu"),
            nn.Conv1D(in_channels=config.num_channels, channels=config.num_channels,
                      kernel_size=config.kernel_size, padding=config.padding),
            nn.BatchNorm(),
            nn.Activation("relu"),
            nn.Conv1D(in_channels=config.num_channels, channels=config.num_channels,
                      kernel_size=config.kernel_size, padding=config.padding)
        )

        """ 定义ShortCut层 """
        self.num_seq = config.max_seq_len
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
            nn.Dropout(config.drop_rate[1]),
            nn.Dense(config.output_size)
        )

    def forward(self, x):
        x = nd.concat(self.embedding(x), self.constant_embedding(x), dim=2)
        x = x.transpose((0,2,1))
        x = self.region_embedding(x)
        x = self.conv_block(x)
        x = self.shortcut_layer(x)
        out = self.output(x)
        return out