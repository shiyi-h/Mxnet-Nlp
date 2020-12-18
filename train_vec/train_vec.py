from collections import Counter
import d2lzh as d2l
import math
from mxnet import autograd, gluon, nd
from mxnet.gluon import data as gdata,loss as gloss,nn
import random
import sys,os
import time
import pickle
import logging
from tqdm import tqdm


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(filename)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

'''读取数据路径'''
load_path = "/home/hh/nlp/data/src"
file_name = "myresult_allinfo_new.pickle 6"

save_path = "/home/hh/nlp/data/res"
save_name = "test_3"



with open(os.path.join(load_path,file_name),"rb") as f:
    dataset = pickle.load(f)
    for st in dataset:
        _type = type(st)
        break
    if _type==str:
        dataset = [st.split() for st in dataset]

class Train_Vec():
    def __init__(self,dataset,batch_size,loss,init_meth,optimizer,embed_size,num_epochs,lr):
        self.batch_size = batch_size
        self.loss = loss
        self.init = init_meth
        self.optimizer = optimizer
        self.lr = lr
        self.embed_size = embed_size
        self.num_epochs = num_epochs
        self.data_init(dataset)


    def data_init(self,dataset):
        self.counter = Counter([tk for st in dataset for tk in st])
        self.idx_to_token = [tk for tk in self.counter]
        self.token_to_idx = {tk: idx for idx, tk in enumerate(self.idx_to_token)}
        self.dataset = [[self.token_to_idx[tk] for tk in st if tk in self.token_to_idx] for st in dataset]
        self.num_tokens = sum([len(st) for st in self.dataset])
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

    @staticmethod
    def skip_gram(center, contexts_and_negatives, embed_v, embed_u):
        v = embed_v(center)
        u = embed_u(contexts_and_negatives)
        pred = nd.batch_dot(v, u.swapaxes(1, 2))
        return pred


    def train(self):
        ctx = d2l.try_gpu()
        init_params = {
            "ctx" : ctx,
            "force_reinit" : True
        }
        if self.init:
            init_params["init"] = self.init
        self.net.initialize(**init_params)
        trainer = gluon.Trainer(self.net.collect_params(), self.optimizer, {"learning_rate": self.lr})
        for epoch in range(self.num_epochs):
            start, l_sum, n = time.time(), 0.0, 0
            for batch in tqdm(self.data_iter,desc="Epoch_{}".format(epoch+1)):
                center, context_negative, mask, label = [data.as_in_context(ctx) for data in batch]
                with autograd.record():
                    pred = self.skip_gram(center, context_negative, self.net[0], self.net[1])
                    l = (self.loss(pred.reshape(label.shape), label, mask) * mask.shape[1] / mask.sum(axis=1))
                l.backward()
                trainer.step(self.batch_size)
                l_sum += l.sum().asscalar()
                n += l.size
            print("epoch %d, loss %.2f, time %.2fs" % (epoch + 1, l_sum / n, time.time() - start))


    def save(self,save_path,save_name):
        w = self.net[0].weight.data()
        with open(os.path.join(save_path,save_name),"wb") as f:
            pickle.dump((w,self.token_to_idx),f)

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
        self.net.add(nn.Embedding(input_dim=len(self.idx_to_token), output_dim=self.embed_size),
                     nn.Embedding(input_dim=len(self.idx_to_token), output_dim=self.embed_size))

        logger.info("Now training....")
        self.train()

if __name__ == '__main__':
    '''基本参数'''
    config = {
        "dataset" : dataset,
        "batch_size" : 512,
        "loss" : gloss.SigmoidBinaryCrossEntropyLoss(),
        "init_meth" : None,
        "optimizer" : "adam",
        "lr" : 0.05,
        "embed_size" : 50,
        "num_epochs" : 5,
    }


    trainer = Train_Vec(**config)
    trainer.run()
    trainer.save(save_path, save_name)