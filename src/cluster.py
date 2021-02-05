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


import pandas as pd
from collections import defaultdict, Counter
from editdistance import distance
from pprint import pprint
import numpy as np
import gensim

DIST = 1

class Graph:
    def __init__(self, data_path, dns_path):
        self.graph = dict() #string -> Node
        self.parent = dict() #string -> parent string
        self.read_data(data_path, dns_path)
        #self.display()

    def read_data(self, data_path, dns_path):
        device_df = pd.read_csv(data_path).fillna('')
        dns_data = pd.read_csv(dns_path).fillna('')

        for i in range(len(device_df)):
            name = device_df['device_vendor'][i]
            device_id = device_df['device_id'][i]
            device_oui = device_df['device_oui'][i]
            disco = device_df['netdisco_device_info'][i]
            dhcp = device_df['dhcp_hostname'][i]
            dns = dns_data[dns_data['device_id'] == device_id]['hostname'].values[0]
            info = [device_oui, dns, dhcp] # ,disco]

            self.connect(name, info)


    def display(self):
        cluster = defaultdict(list)
        for name in self.parent: 
            cluster[self.graph[self.find(name)]] += [self.graph[name]]
        print("cluster number: ", len(cluster))
        ones = 0
        for k, v in sorted(cluster.items(), key=lambda x:-x[0].times):
            print("cluster:", k.name, "times:", k.times)
            ones += k.times==1
            for n in v:
                print(n.name, n.times, end=',') 
            print('\n')
        print(len(cluster), ones)

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
        dictObj = gensim.corpora.Dictionary([node.device_info for node in cluster])
        corpus = [dictObj.doc2bow([node.device_info for node in cluster])
        tfidf = gensim.models.TfidfModel(corpus) 
        for doc in tfidf[corpus]:
            print([[dictObj[id], np.around(freq, decimals=2)] for id, freq in doc])
        


class Node:
    def __init__(self, name, device_info=None, times = 1):
        self.name = name
        self.times = times
        self.device_info = device_info
        #self.preprocess()

    def preprocess(self):
        self.name = self.name.lower() 



