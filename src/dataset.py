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
import numpy as np
from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron

class Dataset:
    def __init__(self, graph):
        self.graph = {k:v for k, v in graph.items() if v.times >= 10 and k not in ('', '?', '??', '???')}


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
        print('data: ', len(train)+len(test))
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


    def bow_softmax(self, feat, percent):
        '''test methods on dns
        '''
        data_t = []; one_hot_X = []; one_hot_y = []
        c = Counter()
        data_oui = []
        for name, node in self.graph.items():
            one_hot_y.append(name)
            for info in node.raw_data: 
                if len(info['device_oui'][0]) != 0 and len(info[feat]) != 0:
                    #print(info)
                    #data_X.update(info[feat])
                    c.update(info[feat])
                    # test on dns+oui
                    c.update(info['device_oui']) 
                    #data_t.append([info[feat], name])
                    data_t.append([info[feat] + info['device_oui'], name])
                    data_oui.append(info['device_oui'][0])
        d = defaultdict(int)
        for k, v in c.items(): 
            if v>=2: 
                one_hot_X.append(k)  
                #d[v] += 1
        #t = sorted(d.items())
        #print(sum(e[1] for e in t))
        X, y = [], []
        dict_X = {e:i for i, e in enumerate(one_hot_X)}
        dict_y = {e:i for i, e in enumerate(one_hot_y)}
        featX_len = len(one_hot_X) ; featy_len = len(one_hot_y)
        for XX, yy in data_t:
            X_t = [0] * featX_len #; y_t = [0] * featy_len 
            for e in XX:
                if e in dict_X: X_t[dict_X[e]] = 1
            #y_t[dict_y[yy]] = 1
            X.append(X_t); y.append(dict_y[yy])
        X = np.array(X); y = np.array(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = percent)
        print(len(X))
        print('Data loaded!')

        #classifier = MultinomialNB()
        classifier = LogisticRegression(C=2,solver='lbfgs',multi_class='multinomial')
        #classifier = Perceptron(tol=1e-3, random_state=0)
        #classifier = xgb.XGBClassifier(max_depth=7, n_estimators=200, colsample_bytree=0.8, 
        #                subsample=0.8, nthread=10, learning_rate=0.1)
        classifier.fit(X_train, y_train)

        y_pred = classifier.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print('\n Accuracy: ', acc)

        # Classification metrics
        #report = classification_report(y_test, y_pred)

        #print('\nClassification Report')
        #print('======================================================')
        #print('\n', report)    
        ret_data = {'model':classifier, 'X':X, 'y':y, 'oui':data_oui ,'dx':dict_X, 'dy':dict_y}
        return ret_data, acc 


    def mix_model(self, dns_data, oui_model):
        ''' mix oui and dns for vendor test
        '''
        #print(model)
        ret = 0
        #missing = 0
        dns_model, data_X, data_y, dict_x, dict_y, data_oui = dns_data['model'], dns_data['X'], dns_data['y'], dns_data['dx'], dns_data['dy'], dns_data['oui']
        #for X, y in zip(data_X, data_y):
        #print(len(data_X))
        dns_prob = dns_model.predict_proba(data_X)
        pred_class = dns_model.predict(data_X)
        #print(prob[0], pred_class[0])
        oui_get = dns_get = 0

        # probe for espressif
        probe = 'Espressif Inc.'
        probe_num = probe_tp = 0
        for i in range(len(data_X)):
            oui_ret = [0, -1]
            oui = data_oui[i]
            #if len(oui) == 0 and len(dns) == 0: missing += 1; continue
            if oui in oui_model:
                name, prob = oui_model[oui][0]
                oui_ret = [prob, dict_y[name]] 
            #prob = dns_model.predict_proba(dns.reshape(-1,1))
            #pred_class = dns_model.predict(dns)
            #print(prob, pred_class)
            #print(len(dns_prob[i]), pred_class[i])
            if oui == probe: 
                probe_num += 1 
                if pred_class[i] < len(dns_prob[i]):
                    ret += pred_class[i] == data_y[i]
                    if pred_class[i] == data_y[i]: probe_tp += 1
                #ret += oui_ret[1] == data_y[i]
                #if oui_ret[1] == data_y[i]: probe_tp+=1
                continue

            if pred_class[i] >= len(dns_prob[i]) or oui_ret[0] >= dns_prob[i][pred_class[i]]:
                ret += oui_ret[1] == data_y[i]
                oui_get += 1
            else: 
                ret += pred_class[i] == data_y[i]
                dns_get += 1
        print(probe_tp / (probe_num+0.001), probe_num)
        return ret / len(data_X) , oui_get, dns_get

    #def port(self, port_data, port_device_data):
    #    pass



    def save_model(self, model,  save_path):
        with open(save_path, 'wb') as fp:
            pickle.dump(model , fp)
        print("model saved!")
    
    def load(self, load_path):
        with open(load_path, 'rb') as fp:
            return pickle.load(fp)


