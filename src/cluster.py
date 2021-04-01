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
# 2. add manual rule

import os
import pandas as pd
import pickle
from collections import defaultdict, Counter
from editdistance import distance
from pprint import pprint
import numpy as np
import gensim
from tabulate import tabulate
import tldextract
from utils import *

DIST = 1
#FIELDS = ['device_vendor', 'device_id', 'device_oui', 'dhcp_hostname' ,'netdisco_device_info', 'dns']
FIELDS = ['device_vendor', 'device_id', 'device_oui', 'dhcp_hostname' ,'netdisco_device_info_list', 'dns', 'port']


class Graph:
    def __init__(self, data_path=None, dns_path=None, port_path=None ,save_path=None, rebuild=False):
        self.graph = dict() #string -> Node
        self.parent = dict() #string -> parent string
        self.device_num = 0
        self.cluster = defaultdict(list)
        self.prob = {}
        self.data_path = data_path
        self.dns_path = dns_path
        self.save_path = save_path
        self.port_path = port_path
        self.rebuild = rebuild
        #self.read_data(data_path, dns_path, save_path, rebuild)
        #self.display()
        '''
        idx = 2
        feat = FIELDS[idx]
        self.graph_prob(feat)
        self.verify(feat, 'tf_idf_veriy')
        '''

    def read_data(self, data_path, dns_path, port_path, save_path, rebuild):
        if not rebuild and os.path.exists(save_path): 
            self.load(save_path) 
            return 
        dns_df = port_df = None #optional
        device_df = pd.read_csv(data_path).fillna('')
        if dns_path: dns_df = pd.read_csv(dns_path).fillna('')
        if port_path: port_df = pd.read_csv(port_path).fillna('')
        raw_data = [] 
        print(len(device_df))
        for i in range(len(device_df)):
            name = manual_rule(device_df[FIELDS[0]][i].lower())
            device_id = device_df[FIELDS[1]][i]
            #device_oui = device_df[FIELDS[2]][i]
            device_oui = get_vendor(device_df[FIELDS[2]][i])
            dhcp = device_df[FIELDS[3]][i].split('-')
            dns= [] ;port= []
            if dns_path: dns = [tldextract.extract(e).domain for e in dns_df[dns_df['device_id'] == device_id]['hostname'].values.tolist()]
            if port_path: port = [i  for e in port_df[port_df['device_id'] == device_id]['port_list'].values.tolist() for i in e.split('+')]
            #print(device_id, dns, sep=';')
            '''
            disco_dict = eval(device_df[FIELDS[4]][i])
            disco = []
            for c in ('name', 'device_type', 'manufacturer'):
                if disco_dict.get(c):
                    disco += [disco_dict.get(c)]
            '''
            disco = []
            info = {FIELDS[2]: [device_oui], FIELDS[3]: dhcp , FIELDS[4]: disco, FIELDS[5]: dns, FIELDS[6]: port} # all values are lists
            #print(name, info)
            print("processing {}-th data".format(i), end="\r") 
            self.connect(name, info)
            self.device_num += 1
            info[FIELDS[0]] = name; info[FIELDS[1]] = device_id
            raw_data += [info]
            #if i == 10: break

        for name in self.parent: 
            self.cluster[self.graph[self.find(name)]] += [self.graph[name]]
        self.save(save_path)
        print("cluster built complete!")
        # return raw data
        return raw_data 

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


    def graph_prob(self, feat):
        graph = {k:v for k, v in self.graph.items() if v.times >= 10 and k not in ('', '?', '??', '???')}
        #device_num = sum(graph[node].times for node in graph)
        device_num = sum(node.feat_cnt[feat] for _, node in graph.items()) 
        p_device = {name: node.feat_cnt[feat] / device_num  for name, node in graph.items()}  
        num_feat = defaultdict(int)
        for _, node in graph.items():
            for k, v in node.device_info[feat].items(): 
                num_feat[k] += v
        num_all = sum(v for k, v in num_feat.items() )
        p_feat = {k: v/ num_all for k, v in num_feat.items() }
        prob = defaultdict(defaultdict)
        for device in p_device:   
            cond = graph[device].prob(feat)
            for f_name in p_feat:   
                prob[f_name][device] = cond[device][f_name] * p_device[device] / p_feat[f_name] # Bayes 
        
        for k in prob:
            prob[k] = sorted(prob[k].items(), key=lambda x: -x[1])
        self.feat2device = prob
        #pprint(prob)
        #print('----------')
        #pprint( sum(v for k,v in p_feat.items()))
        #pprint( sum(v for k,v in p_device.items()))

    def tf_idf_veriy(self, feat , center, cluster):  # for one cluster
        '''Use tf-idf on oui, dns, dhcp, disco to verify one cluster
           Every node is a document, the whole cluster is a corpus
        '''
        #name = [node.name for node in cluster if len(node.device_info[feat])!=0]
        #dictObj = gensim.corpora.Dictionary([gensim.utils.simple_preprocess(node.device_info, min_len=4) for node in cluster])

        docu = []; name = []
        for node in cluster:
            cur = []
            if len(node.device_info[feat]) != 0: name += [node.name]
            for k, v in node.device_info[feat].items():
                cur += [k]*v
            docu.append(cur)

        dictObj = gensim.corpora.Dictionary(docu)
        corpus = [dictObj.doc2bow(e) for e in docu]
        #corpus = [dictObj.doc2bow([k]*v) for node in cluster for k, v in node.device_info[feat].items()]
        #corpus = [dictObj.doc2bow(gensim.utils.simple_preprocess(node.device_info, min_len=4)) for node in cluster]
        tfidf = gensim.models.TfidfModel(corpus) 
        #print(name, len(name), len(tfidf[corpus]), len(corpus))
        for doc_idx in range(len(name)):
            doc = tfidf[corpus[doc_idx]]
            print(name[doc_idx], end=":")
            for id, freq in sorted(doc, key=lambda x:-x[1])[:5]:
                print(dictObj[id], np.around(freq, decimals=3), end=',')
            print('\n')
        return []

    def bayes_verify(self, feat, center, cluster):  #for one cluster,{center: cluster}
        for node in cluster:
            info = node.device_info[feat].most_common(2)
            if len(info) == 0: return []
            info = info[1][0] if not info[0][0] and len(info)==2 else info[0][0]
            table = []
            if self.feat2device[info] and center.name != self.feat2device[info][0][0] and (len(self.cluster[center])==1 or center.name != node.name):
                table += [[ center.name,  node.name, node.times , info , self.feat2device[info][0] ]] 
            return table
    
    def verify(self, feat, method):
        '''method: bayes & tf-idf
        '''
        print('verifying!')
        table = []
        for center in self.cluster:  
            #table += self.bayes_verify(feat, center,  self.cluster[center])
            table += getattr(self, method)(feat, center,  self.cluster[center])
        print(tabulate(table, headers= ["cluster","node","times", "info", "infer"]))



    def save(self, save_path): 
        with open(save_path, 'wb') as fp:
            pickle.dump({'cluster': self.cluster, 'graph': self.graph, 'parent':self.parent} , fp)
        print("cluster saved!")

    def load(self, load_path):
        with open(load_path, 'rb') as fp:
            data = pickle.load(fp)
            self.cluster, self.graph, self.parent = data['cluster'], data['graph'], data['parent']
        print("cluster loaded!")


class Node:
    def __init__(self, name, device_info=None, times = 1):
        self.name = name
        self.times = times
        self.device_info = {k: Counter() for k in device_info}
        self.feat_cnt = Counter()
        self.raw_data = [] # for dns mainly 
        self.add_info(device_info)

    def add_info(self, device_info):
        #print(device_info)
        self.raw_data.append(device_info)
        for k, v in device_info.items():
            self.device_info[k].update(v)
            self.feat_cnt[k] += len(v)

    def prob(self, feat):
        num = self.feat_cnt[feat]
        #num = sum(v for k, v in self.device_info[feat].items())  
        #num = sum(v for k, v in self.device_info[feat].items() if k != '')  
        p_cond = defaultdict(int)
        for k, v in self.device_info[feat].items():
            #if k != '': p_cond[k] =  v / num   
            p_cond[k] =  v / num   
        ret = {}
        ret[self.name] = p_cond
        #print(self.name, [(e[0], round(e[1], 2)) for e in sorted(p_cond.items(), key=lambda x: -x[1])])
        return ret




