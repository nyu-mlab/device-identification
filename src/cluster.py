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

class Graph:
    def __init__(self, data_path, dns_path, save_path, rebuild=False):
        self.graph = dict() #string -> Node
        self.parent = dict() #string -> parent string
        self.cluster = defaultdict(list)
        self.read_data(data_path, dns_path, save_path, rebuild)
        self.display()

    def read_data(self, data_path, dns_path, save_path, rebuild):
        if not rebuild and os.path.exists(save_path): 
            self.load(save_path) 
            return 
        device_df = pd.read_csv(data_path).fillna('')
        dns_data = pd.read_csv(dns_path).fillna('')

        for i in range(len(device_df)):
            name = device_df['device_vendor'][i]
            device_id = device_df['device_id'][i]
            device_oui = device_df['device_oui'][i]
            disco = device_df['netdisco_device_info'][i]
            dhcp = device_df['dhcp_hostname'][i]
            dns = dns_data[dns_data['device_id'] == device_id]['hostname'].values.tolist()
            #print(device_id, dns, sep=';')
            info = [device_oui] + dns +  [dhcp] # ,disco]
            #print(info)
            print("processing {}-th data".format(i), end="\r") 
            self.connect(name, info)

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
            self.verify(v)
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
            self.graph[name].device_info += info
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
        dictObj = gensim.corpora.Dictionary([node.device_info for node in cluster])
        corpus = [dictObj.doc2bow(node.device_info) for node in cluster]
        tfidf = gensim.models.TfidfModel(corpus) 
        for doc in tfidf[corpus]:
            print([[dictObj[id], np.around(freq, decimals=2)] for id, freq in doc])
        

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
        self.device_info = device_info
        #self.preprocess()

    def preprocess(self):
        self.name = self.name.lower() 



