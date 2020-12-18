import warnings
warnings.filterwarnings('ignore')
import itertools
import time
import logging
import os
import mxnet as mx
from mxnet.gluon import data as gdata
from mxnet import autograd, nd, init
import gluonnlp as nlp
import pickle
from tqdm import tqdm
from word_embedding import model as nlpmodel, data as nlpdata
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(filename)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)



'''读取数据路径'''
load_path = "/home/hh/nlp/data/src"
file_name = "full_out.txt"

save_path = "/home/hh/nlp/data/res"
save_name = "007.std.300d.vec"



class Train_Vec():
    def __init__(self,load_path,file_name,num_epochs,batch_size,embed_size,lr,ctx,
                 init_meth=init.Uniform(),min_freq=1,cbow=False, large=False):
        self.num_epochs = num_epochs
        self.ctx = ctx
        self.embed_size = embed_size
        self.batch_size = batch_size
        self.cbow = cbow
        self.lr = lr
        self.init = init_meth
        if large:
            self.large_init(load_path, file_name, min_freq)
        else:
            self.data_init(load_path,file_name,min_freq)

    @staticmethod
    def read_iter(load_path, file_name):
        lines = []
        break_mark = 0
        with open(os.path.join(load_path,file_name), "r") as f:
            while True:
                line = f.readline().split()
                if not line:
                    break_mark += 1
                    if break_mark == 5:
                        break
                    else:
                        continue
                lines.append(line)
                if len(lines) >= 1000000:
                    yield lines
                    lines.clear()
        if lines:
            yield lines

    def large_init(self, load_path, file_name, min_freq, total_init=100):
        def _get_counter(data):
            counter = {}
            t = tqdm(total=total_init, desc="get_counter")
            for d in data:
                c = nlp.data.count_tokens(itertools.chain.from_iterable(d))
                for key in c:
                    if key in counter:
                        counter[key] += c[key]
                    else:
                        counter[key] = c[key]
                t.update(1)
            t.close()
            print("get counter done...")
            return counter
        def code(sentence):
            return [vocab[token] for token in sentence if token in vocab]
        counter = _get_counter(self.read_iter(load_path, file_name))
        vocab = nlp.Vocab(counter, unknown_token=None, padding_token=None,
                          bos_token=None, eos_token=None, min_freq=min_freq)
        self.idx_to_counts = [counter[w] for w in vocab.idx_to_token]
        total_data = self.read_iter(load_path, file_name)
        data = []
        t = tqdm(total=total_init, desc="set_idx_data")
        for d in total_data:
            d = [code(s) for s in d]
            data += d
            t.update(1)
        t.close()
        print("Reset data to idx_data done...")
        data = nlp.data.SimpleDataStream([data])
        self.vocab = vocab
        self.data, batchify_fn, self.subword_function = nlpdata.transform_data_fasttext(
            data, self.vocab, self.idx_to_counts, cbow=self.cbow, ngrams=[3, 4, 5, 6], ngram_buckets=100000,
            batch_size=self.batch_size, window_size=5)
        self.batches = self.data.transform(batchify_fn)


    def data_init(self,load_path,file_name,min_freq):
        data = nlp.data.CorpusDataset(os.path.join(load_path,file_name))
        counter = nlp.data.count_tokens(itertools.chain.from_iterable(data))
        vocab = nlp.Vocab(counter, unknown_token=None, padding_token=None,
                          bos_token=None, eos_token=None, min_freq=min_freq)
        self.idx_to_counts = [counter[w] for w in vocab.idx_to_token]
        def code(sentence):
            return [vocab[token] for token in sentence if token in vocab]
        data = data.transform(code, lazy=False)
        data = nlp.data.SimpleDataStream([data])
        self.vocab = vocab
        self.data, batchify_fn, self.subword_function = nlpdata.transform_data_fasttext(
            data, self.vocab, self.idx_to_counts, cbow=self.cbow, ngrams=[3, 4, 5, 6], ngram_buckets=100000,
            batch_size=self.batch_size, window_size=5)
        self.batches = self.data.transform(batchify_fn)

    def train_init(self):
        negatives_weights = nd.array(self.idx_to_counts)
        model = nlpmodel.CBOW if self.cbow else nlpmodel.SG
        self.net = model(
            self.vocab.token_to_idx, self.embed_size, self.batch_size, negatives_weights, self.subword_function,
            num_negatives=5, smoothing=0.75)
        self.net.initialize(init=self.init, ctx=self.ctx)
        self.net.hybridize()
        self.trainer = mx.gluon.Trainer(self.net.collect_params(), 'adagrad', dict(learning_rate=self.lr))


    def train_embedding(self):
        self.train_init()
        log_interval = 500
        for epoch in range(1, self.num_epochs + 1):
            start = time.time()
            l_avg = 0
            print('Beginning epoch %d and resampling data.' % epoch)
            for i,batch in enumerate(self.batches):
                batch = [array.as_in_context(self.ctx) for array in batch]
                with autograd.record():
                    l = self.net(*batch)
                l.backward()
                self.trainer.step(1)
                l_avg += l.mean()
                if i % log_interval == 0 and i!=0:
                    nd.waitall()
                    l_avg = l_avg.asscalar() / log_interval
                    print('epoch %d, iteration %d, loss %.2f, use time %.3f sec'
                          % (epoch, i, l_avg, time.time()-start))
                    start = time.time()
                    l_avg = 0

    def save(self,save_path,save_name):
        logger.info("Now saving vec...")
        with open(os.path.join(save_path,save_name),"a") as f:
            f.write("%s %s" %(len(self.vocab.idx_to_token), self.embed_size))
            for tk in self.vocab.idx_to_token:
                f.write("\n"+tk+" "+" ".join(list(map(str, self.net[tk].asnumpy()))))
        filename = os.path.join(save_path,save_name+".params")
        self.net.save_parameters(filename)

if __name__ == '__main__':
    '''基本参数'''
    config = {
        "load_path" : load_path,
        "file_name" : file_name,
        "batch_size" : 512,
        "lr" : 0.05,
        "embed_size" : 300,
        "num_epochs" : 100,
        "init_meth" : init.Xavier(),
        "ctx" : mx.gpu(1),
        "min_freq" : 1,
        "cbow" : False,
        "large" : True
    }
    trainer = Train_Vec(**config)
    trainer.train_embedding()
    trainer.save(save_path,save_name)

