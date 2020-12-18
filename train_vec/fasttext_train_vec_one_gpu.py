from collections import Counter
import d2lzh as d2l
import math
from mxnet import autograd, gluon, nd, init
from mxnet.gluon import data as gdata,loss as gloss,nn, utils as gutils
import random
import sys,os
import time
import pickle
import logging
from tqdm import tqdm
sys.path.append("../")
from tools import fnv
import mxnet as mx


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(filename)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

'''读取数据路径'''
load_path = "/home/hh/nlp/data/src"
file_name = "sum_std_info_Two.txt"

save_path = "/home/hh/nlp/data/res"
save_name = "007.std2.300d.vec"


with open(os.path.join(load_path,file_name),"r") as f:
    dataset = f.readlines()
    for st in dataset:
        _type = type(st)
        break
    if _type==str:
        dataset = [st.split() for st in dataset]

class Train_Vec():
    def __init__(self,dataset,batch_size,loss,init_meth,optimizer,embed_size,num_epochs,lr,n_grams_num,ctx):
        self.batch_size = batch_size
        self.loss = loss
        self.init = init_meth
        self.optimizer = optimizer
        self.lr = lr
        self.embed_size = embed_size
        self.num_epochs = num_epochs
        self.n_grams_num = n_grams_num
        self.ctx = ctx
        self.data_init(dataset)

    def n_grams(self,x):
        x = "<"+x+">"
        lst = [x]
        for i in range(len(x)):
            if i+self.n_grams_num>len(x): break
            lst.append(fnv.hash(x[i:self.n_grams_num+i],algorithm=fnv.fnv_1a,bits=64))
        return lst


    def data_init(self,dataset):
        self.counter = Counter([tk for st in dataset for tk in st])
        self.idx_to_token = [tk for tk in self.counter]
        self.token_to_idx = {tk: idx for idx, tk in enumerate(self.idx_to_token)}
        self.dataset = [[self.token_to_idx[tk] for tk in st if tk in self.token_to_idx] for st in dataset]
        self.num_tokens = sum([len(st) for st in self.dataset])
        n_grams_set = set()
        self.center_n_grams = {}
        for tk in self.idx_to_token:
            tk_n_grams = self.n_grams(tk)
            n_grams_set |= set(tk_n_grams)
            self.center_n_grams[self.token_to_idx[tk]] = tk_n_grams
        n_grams_set = {gram:idx for idx,gram in enumerate(n_grams_set)}
        self.n_grams_set_size = len(n_grams_set)
        for tk_idx in self.center_n_grams:
            self.center_n_grams[tk_idx] = [n_grams_set[gram] for gram in self.center_n_grams[tk_idx]]
        logger.info("Data initializa Done...")

    def discard(self,idx, t=1e-4):
        return random.uniform(0, 1) < 1 - math.sqrt(t / self.counter[self.idx_to_token[idx]] * self.num_tokens)

    def get_centers_and_contexts(self, dataset, max_window_size):
        centers, contexts = [], []
        for st in dataset:
            if len(st) < 2: continue
            centers += st
            for center_i in range(len(st)):
                window_size = random.randint(1, max_window_size)
                indices = list(range(max(0, center_i - window_size), min(len(st), center_i + 1 + window_size)))
                indices.remove(center_i)
                contexts.append([st[idx] for idx in indices])
        return centers, contexts

    @staticmethod
    def get_negatives(all_contexts, sampling_weights, k):
        all_negatives, neg_candidates, i = [], [], 0
        population = list(range(len(sampling_weights)))
        for contexts in all_contexts:
            negatives = []
            while len(negatives) < len(contexts) * k:
                if i == len(neg_candidates):
                    i, neg_candidates = 0, random.choices(population, sampling_weights, k=int(1e5))
                neg, i = neg_candidates[i], i + 1
                if neg not in set(contexts):
                    negatives.append(neg)
            all_negatives.append(negatives)
        return all_negatives

    @staticmethod
    def batchify(data):
        max_len = max(len(c) + len(n) for _, c, n in data)
        centers, contexts_negatives, masks, labels = [], [], [], []
        for center, context, negative in data:
            cur_len = len(context) + len(negative)
            centers += [center]
            contexts_negatives += [context + negative + [0] * (max_len - cur_len)]
            masks += [[1] * cur_len + [0] * (max_len - cur_len)]
            labels += [[1] * len(context) + [0] * (max_len - len(context))]
        return (nd.array(centers).reshape((-1, 1)), nd.array(contexts_negatives), nd.array(masks), nd.array(labels))



    def Dataload(self,all_centers, all_contexts, all_negatives):
        num_workers = 0 if sys.platform.startswith("win32") else 4
        dataset = gdata.ArrayDataset(all_centers, all_contexts, all_negatives)
        self.data_iter = gdata.DataLoader(dataset, self.batch_size, shuffle=True, batchify_fn=self.batchify, num_workers=num_workers)
        for batch in self.data_iter:
            for name, data in zip(["centers", "contexts_negatives", "masks", "labels"], batch):
                print(name, "shape:", data.shape)
            break

    def skip_gram(self,center, contexts_and_negatives, embed_v, embed_u):
        v = [embed_v(nd.array(self.center_n_grams[i.asscalar()]).as_in_context(center.context)).sum(axis=0).reshape(
            (1,1,-1)) for i in center]
        v = nd.concat(*v,dim=0)
        u = embed_u(contexts_and_negatives)
        pred = nd.batch_dot(v, u.swapaxes(1, 2))
        return pred

    @staticmethod
    def _get_batch(batch, ctx):
        """Return features and labels on ctx."""
        center, context_negative, mask, label = batch
        for field in [mask, label]:
            if field.dtype != context_negative.dtype:
                field[:] = field.astype(context_negative.dtype)
        return (gutils.split_and_load(center, even_split=False,ctx_list=ctx),
                gutils.split_and_load(context_negative, even_split=False,ctx_list=ctx),
                gutils.split_and_load(mask, even_split=False,ctx_list=ctx),
                gutils.split_and_load(label, even_split=False,ctx_list=ctx), center.shape[0])

    def train(self):
        print('training on', self.ctx)
        init_params = {
            "ctx" : self.ctx,
            "force_reinit" : True
        }
        if self.init:
            init_params["init"] = self.init
        self.net.initialize(**init_params)
        trainer = gluon.Trainer(self.net.collect_params(), self.optimizer, {"learning_rate": self.lr})
        for epoch in range(self.num_epochs):
            start, l_sum, n = time.time(), 0.0, 0
            for batch in tqdm(self.data_iter, desc="Epoch_{}".format(epoch + 1)):
                center, context_negative, mask, label = [data.as_in_context(self.ctx) for data in batch]
                with autograd.record():
                    pred = self.skip_gram(center, context_negative, self.net[0], self.net[1])
                    l = (self.loss(pred.reshape(label.shape), label, mask) * mask.shape[1] / mask.sum(axis=1))
                l.backward()
                trainer.step(self.batch_size)
                l_sum += l.sum().asscalar()
                n += l.size
            print("epoch %d, loss %.2f, time %.2fs" % (epoch + 1, l_sum / n, time.time() - start))


    def save(self,save_path,save_name):
        w = self.net[0].weight.data(self.ctx)
        logging.info("Now saving vec...")
        with open(os.path.join(save_path,save_name),"a") as f:
            f.write("%s %s" %(len(self.token_to_idx), self.embed_size))
            for tk in self.token_to_idx:
                f.write("\n"+tk+" "+" ".join(list(map(str, w[self.token_to_idx[tk]].asnumpy()))))

    def run(self):
        subsampled_dataset = [[tk for tk in st if not self.discard(tk)] for st in self.dataset]
        all_centers, all_contexts = self.get_centers_and_contexts(subsampled_dataset, 5)
        logger.info("Extract centers,contexts Done...")

        # 根据word2vec论文建议,噪声词采样概率设为w词频与总词频之比的0.75次方
        sampling_weights = [self.counter[w] ** 0.75 for w in self.idx_to_token]
        all_negatives = self.get_negatives(all_contexts, sampling_weights, 5)
        logger.info("Create negatives Done...")

        self.Dataload(all_centers, all_contexts, all_negatives)

        self.net = nn.Sequential()
        self.net.add(nn.Embedding(input_dim=self.n_grams_set_size, output_dim=self.embed_size),
                     nn.Embedding(input_dim=len(self.idx_to_token), output_dim=self.embed_size))

        logger.info("Now training....")
        self.train()

if __name__ == '__main__':
    '''基本参数'''
    config = {
        "dataset" : dataset,
        "batch_size" : 512,
        "loss" : gloss.SigmoidBinaryCrossEntropyLoss(),
        "init_meth" : init.Normal(0.01),
        "optimizer" : "adam",
        "lr" : 0.001,
        "embed_size" : 100,
        "num_epochs" : 10,
        "n_grams_num" : 3,
        "ctx" : d2l.try_gpu()
    }

    trainer = Train_Vec(**config)
    trainer.run()
    trainer.save(save_path, save_name)

