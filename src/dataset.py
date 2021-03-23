'''
Tom Long@mLab
03/09/2021

Dataset to train and test different methods
Provide data with gt, which appear for many times

03/22: Try BoW with softmax

'''

import random
import gensim
import pickle
from collections import defaultdict, Counter


class Dataset:
    def __init__(self, graph):
        self.graph = {k:v for k, v in graph.items() if v.times >= 5 and k not in ('', '?', '??', '???')}


    def train(self, data, method):
        '''return trained model
        '''
        print("Training!")
        return getattr(self, method)(data, 'train') 


    def test(self, data, method, model):
        '''test model using method on data
        '''
        print("Testing!")
        return getattr(self, method)(data, 'test', model) 


    def bayes(self, data, state, model=None):
        '''state: train or test
        '''
        if state == 'train':
            device = set([e[0] for e in data])
            feat = set([e[1] for e in data])
            model = defaultdict(defaultdict)
            for f in feat:
                for d in device:
                    model[f][d] = 0
            for d, f in data:
                model[f][d] += 1
            for f in feat:
                n = sum(v for k,v in model[f].items())
                for d in device:
                    model[f][d] /= n
                model[f] = sorted(model[f].items(), key=lambda x: -x[1])
            return model 

        if state == 'test':
            ret = 0
            missing = 0
            tp = Counter()
            for d, f in data:
                if f not in model:
                    missing += 1
                elif model[f][0][0] == d: 
                    ret += 1 
                    tp.update([d])
            #print(tp)
            return ret / (len(data) - missing), tp

    def tf_idf(self, data, state, model=None):
        if state == 'train':
            device = set()
            docu = []
            temp = defaultdict(list) 
            for d, f in data:
                temp[d] += [f]
                device.add(d)
            device = list(device)
            for n in device:
                docu += [temp[n]]
            dictObj = gensim.corpora.Dictionary(docu)
            corpus = [dictObj.doc2bow(e) for e in docu]
            tfidf = gensim.models.TfidfModel(corpus) 
            mapping = defaultdict(list)
            for doc_idx in range(len(device)):
                doc = tfidf[corpus[doc_idx]]
                #print(name[doc_idx], end=":")
                for idd, freq in sorted(doc, key=lambda x:-x[1])[:3]:
                    cur = dictObj[idd]
                    mapping[cur] += [(device[doc_idx], freq)]
                    #if cur not in mapping or cur in mapping and mapping[cur][1] < freq :
                    #    mapping[cur] = (device[doc_idx], freq) 
            for k in mapping:
                mapping[k] = sorted(mapping[k], key=lambda x:-x[1])
            return mapping

        if state == 'test':
            #print(model)
            ret = 0
            missing = 0
            tp = Counter()
            for d, f in data:
                if f not in model:
                    missing += 1
                elif model[f][0][0] == d: 
                    ret += 1 
                    tp.update([d])
            #print(tp)
            return ret / (len(data) - missing), tp
             

    def train_test(self, feat, method, percent):
        self.split_data(feat, percent)
        train, test = self.trainData, self.testData
        model = self.train(train, method)
        ret, tp = self.test(test, method, model)
        print("Using {} on {}, acc: {}.".format(method, feat, ret))
        return model, ret

    def split_data(self, feat, percent):
        '''percent: percentage of training data
        '''
        data = []
        for name, node in self.graph.items():
            for d, t in node.device_info[feat].items():
                if d == '': continue
                data += [[name, d]]*t 
        random.shuffle(data)
        n = round(len(data) * percent)
        self.trainData, self.testData = data[:n], data[n:]

    def save_model(self, model,  save_path):
        with open(save_path, 'wb') as fp:
            pickle.dump(model , fp)
        print("model saved!")
