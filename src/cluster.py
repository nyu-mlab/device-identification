# Taiyu Long@mLab
# 01/25/2021 


# Design:
# Node: name, times
# Graph: Node, DSU, data stream
# connect nodes by distance
# name a cluster with the name occurs most
# time: O(n2), space: O(n)
# implementation: Disjoint set union
# times as ranking

# optimization:
# 1. when new data comes, match it with cluster by their ranking

import os
import pandas as pd
import pickle
from collections import defaultdict, Counter
from editdistance import distance
from pprint import pprint
import numpy as np
import gensim

DIST = 1
FIELDS = ['device_vendor', 'device_id', 'device_oui', 'dhcp_hostname' ,'netdisco_device_info']


class Graph:
    def __init__(self, data_path, dns_path, save_path, rebuild=False):
        self.graph = dict() #string -> Node
        self.parent = dict() #string -> parent string
        self.device_num = 0
        self.cluster = defaultdict(list)
        self.prob = {}
        self.read_data(data_path, dns_path, save_path, rebuild)
        #self.display()
        self.graph_prob()

    def read_data(self, data_path, dns_path, save_path, rebuild):
        if not rebuild and os.path.exists(save_path): 
            self.load(save_path) 
            return 
        device_df = pd.read_csv(data_path).fillna('')
        dns_data = pd.read_csv(dns_path).fillna('')
        
        for i in range(len(device_df)):
            name = device_df[FIELDS[0]][i]
            device_id = device_df[FIELDS[1]][i]
            device_oui = device_df[FIELDS[2]][i]
            dhcp = device_df[FIELDS[3]][i]
            dns = dns_data[dns_data['device_id'] == device_id]['hostname'].values.tolist()
            #print(device_id, dns, sep=';')
            disco_dict = eval(device_df[FIELDS[4]][i])
            disco = []
            for c in ('name', 'device_type', 'manufacturer'):
                if disco_dict.get(c):
                    disco += [disco_dict.get(c)]
            info = {FIELDS[2]: device_oui, FIELDS[3]: dhcp + FIELDS[4]:disco , 'dns': dns} 
            #print(info)
            print("processing {}-th data".format(i), end="\r") 
            self.connect(name, info)
            self.device_num += 1

        for name in self.parent: 
            self.cluster[self.graph[self.find(name)]] += [self.graph[name]]
        self.save(save_path)
        print("cluster built complete!")

    def display(self):
        cluster = self.cluster
        print("cluster number: ", len(cluster))
        ones = 0
        for k, v in sorted(cluster.items(), key=lambda x:-x[0].times):
            print("cluster:", k.name, "times:", k.times)
            #self.verify(v)
            ones += k.times==1
            for n in v:
                print(n.name, n.times, end=',') 
            print('\n')
        #print(len(cluster), ones)

    def find(self, name):
        if name != self.parent[name]:
            self.parent[name] = self.find(self.parent[name]) 
        return self.parent[name]


    def union(self, name1, name2):
        p1, p2 = self.find(name1), self.find(name2) 
        if p1 == p2: return 
        if self.graph[p1].times >= self.graph[p2].times:
            p1, p2 = p2, p1
        self.parent[p1] = p2


    def connect(self, name, info):
        name = name.lower()
        if name in self.graph:
            self.graph[name].times += 1
            self.graph[name].add_info(info)
            parent = self.find(name)
            self.parent[name] = name
            self.union(name, parent) 
        else:
            newNode = Node(name, info)
            self.graph[name] = newNode
            self.parent[name] = name 
            for name_ in self.graph:
                if distance(name, name_) <= DIST:
                    self.union(name, name_)
                    return 

                    
    def verify(self, cluster):
        '''Use tf-idf on oui, dns, dhcp, disco to verify one cluster
           Every node is a document, the whole cluster is a corpus
        '''
        #print("verifying !")
        name = [node.name for node in cluster]
        dictObj = gensim.corpora.Dictionary([gensim.utils.simple_preprocess(node.device_info, min_len=4) for node in cluster])
        corpus = [dictObj.doc2bow(gensim.utils.simple_preprocess(node.device_info, min_len=4)) for node in cluster]
        tfidf = gensim.models.TfidfModel(corpus) 
        for doc_idx in range(len(name)):
            doc = tfidf[corpus][doc_idx]
            print(name[doc_idx], end=":")
            for id, freq in sorted(doc, key=lambda x:-x[1])[:5]:
                print(dictObj[id], np.around(freq, decimals=2), end=',')
            print('\n')

    def graph_prob(self, feat):
        p_device = {name: node.times / self.device_num for name, node in self.graph.items()}  
        num_feat = defaultdict(int)
        for _, node in self.graph.items():
            for k, v in node.device_info[feat]: 
                num_feat[k] += v
        num_all = sum(v for k, v in num_feat.items() if k!= '')
        p_feat = {k: v/ num_all for k, v in num_feat.items() if k!= ''}
        prob = defaultdict(defaultdict(float))
        for device in p_device:   
            for feat in p_feat:   
                cond = self.graph[device].prob(feat)
                prob[feat][device] = cond[device][feat] * p_device[device] / p_feat[feat] # Bayes 
        
        pprint(prob)

    def save(self, save_path): 
        with open(save_path, 'wb') as fp:
            pickle.dump(self.cluster, fp)
        print("cluster saved!")

    def load(self, load_path):
        with open(load_path, 'rb') as fp:
            self.cluster = pickle.load(fp)
        print("cluster loaded!")


class Node:
    def __init__(self, name, device_info=None, times = 1):
        self.name = name
        self.times = times
        self.device_info = {k: Counter() for k in device_info}
        self.add_info(device_info)
        #self.preprocess()

    def preprocess(self):
        self.name = self.name.lower() 

    def add_info(self, device_info):
        for k, v in device_info:
            self.device_info[k].update(v)

    def prob(self, feat):
        num = sum(v for k, v in self.device_info[feat] if k != '')  
        p_cond = {k: v / num for k, v in self.device_info[feat] if k != ''}  
        return {self.name: p_cond}




