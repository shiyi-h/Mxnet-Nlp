import os
import mxnet as mx
from mxnet.gluon import loss as gloss
from .focal_loss import FocalLoss


class Config():
    """配置参数"""
    def __init__(self,base_path, model, data, vec, seq_len, target, num_classes):
        self.ctx = [mx.gpu(4),mx.gpu(5),mx.gpu(6),mx.gpu(7)]
        self.max_seq_len = seq_len
        self.target = target
        self.kfold_num = 5
        self.data_path = os.path.join(base_path,"data/"+data+"/data.txt")
        self.vec_path = os.path.join(base_path, "data/" + data, vec)
        data_base_path = os.path.join(base_path, "data", data, target)
        if not os.path.exists(data_base_path):
            os.mkdir(data_base_path)
        self.params_path = os.path.join(data_base_path,"params.pk")
        self.std_path = os.path.join(data_base_path,"std.pk")
        self.train_path = os.path.join(data_base_path,"train")
        self.dev_path = os.path.join(data_base_path,"dev")
        self.test_path = os.path.join(data_base_path,"test")
        for path in [self.train_path,self.dev_path,self.test_path]:
            if not os.path.exists(path):
                os.makedirs(path)
        self.save_model_name = model
        save_base_path = os.path.join(base_path,"model_save",data,target)
        if not os.path.exists(save_base_path):
            os.makedirs(save_base_path)
        self.save_path = os.path.join(save_base_path,self.save_model_name+".params")
        self.require_improvement = 2000 # 若超过这个数量的batch效果还未提升, 提前结束训练
        self.print_epoch_batch = 1000 # 每个epoch经过多少次batch输出一次
        self.embed_size = 300
        self.output_size = num_classes
        self.num_epochs = 20
        self.batch_size = 256
        self.lr = 5e-5
        self.optimizer = "adam"
        self.wd = 0.00
        self.init = mx.init.Xavier()
        self.loss = gloss.SoftmaxCrossEntropyLoss()