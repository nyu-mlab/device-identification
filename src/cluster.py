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

DIST = 1

class Graph:
    def __init__(self, data_path):
        self.graph = dict() #string -> Node
        self.parent = dict() #string -> parent string
        self.read_data(data_path)
        self.display()

    def read_data(self, data_path):
        device_df = pd.read_csv(data_path).fillna('')
        fields = device_df.columns[1:5]
        for name in device_df[fields[2]]:
            self.connect(name)


    def display(self):
        cluster = defaultdict(list)
        for name in self.parent: 
            cluster[self.graph[self.find(name)]] += [self.graph[name]]
        print("cluster number: ", len(cluster))
        for k, v in sorted(cluster.items(), key=lambda x:-x[0].times):
            print("cluster:", k.name, "times:", k.times)
            for n in v:
                print(n.name, n.times, end=',') 
            print('\n')


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


    def connect(self, name):
        name = name.lower()
        if name in self.graph:
            self.graph[name].times += 1
            parent = self.find(name)
            self.parent[name] = name
            self.union(name, parent) 
        else:
            newNode = Node(name)
            self.graph[name] = newNode
            self.parent[name] = name 
            for name_ in self.graph:
                if distance(name, name_) <= DIST:
                    #self.graph[name].neib += [newNode]
                    #newNode.neib += [self.graph[name]]
                    self.union(name, name_)
                    return 

                    

class Node:
    def __init__(self, name, times = 1):
        self.name = name
        self.times = times
        #self.preprocess()
        #self.neib = [] 

    def preprocess(self):
        self.name = self.name.lower() 



