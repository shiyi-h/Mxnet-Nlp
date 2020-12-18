from train_eval import train
import time
import argparse
from importlib import import_module
from utils import build_dataset,bulid_dataset_kfold
import warnings
warnings.filterwarnings(category=UserWarning,action="ignore")

parser = argparse.ArgumentParser(description='007 standard_label predict')
parser.add_argument('--model', type=str, required=True,
                    help='choose a model: Bert, DPCNN, TextCNN, LSTM, LSTMCNN, TextDPCNN, TextIncDPCNN')
parser.add_argument('--prefix', type=str, default="/home/hh/nlp/bert_pred_std", help='input the data path for train')
parser.add_argument('--vec', type=str, default="007.std.300d.vec", help='input the data path for train')
parser.add_argument('--data', type=str, required=True, help='input the name of data')
parser.add_argument('--target', type=str, required=True, help='input the target of data')
args = parser.parse_args()

if __name__ == '__main__':
    data = args.data
    model_name = args.model
    prefix = args.prefix
    vec = args.vec
    target = args.target
    seq_len = 50 if data=="user_split" else 50
    if target=="standard_label":
        num_classes = 1101 if data=="user_split" else 1086
    else:
        num_classes = 79 if data=="user_split" else 397
    print(prefix)
    m = import_module('models.' + model_name)
    config = m.Config(prefix, model_name, data, vec, seq_len, target, num_classes)
    start_time = time.time()
    print("Loading data...")
    train_iter, dev_iter = bulid_dataset_kfold(config)

    print("Time usage : %.3f" %(time.time()-start_time))

    # train
    model = m.Model(config)
    train(config, model, train_iter, dev_iter, test_iter=False)


