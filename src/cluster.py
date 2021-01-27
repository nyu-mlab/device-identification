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
from collections import defaultdict
from editdistance import distance

DATA_PATH = '../data/device.csv'
DIST = 1

class Graph:
    def __init__(self, data_path):
        self.read_data(data_path)
        self.graph = dict() #string -> Node
        self.parent = dict()

    def read_data(self, data_path):
        device_df = pd.read_csv(data_path).fillna('')
        fields = device_df.columns[1:5]
        

    def find(self, node):
        if node != self.parent[node]: self.parent[node] = self.find(self.parent[node]) 
        return self.parent[node]

    def union(self, node1, node2):
        p1, p2 = self.find(node1), self.find(node2) 
        if p1.times > p2.times:
            p1, p2 = p2, p1
        self.parent[p1] = p2

    def connect(self, name):
        if name in self.graph:
            self.graph[name].times += 1
        else:
            newNode = Node(name)
            self.graph[name] = newNode
            self.parent[newNode] = newNode
            for name_ in self.graph:
                if distance(name, name_) <= DIST:
                    #self.graph[name].neib += [newNode]
                    #newNode.neib += [self.graph[name]]
                    self.union(newNode, self.graph[name_])
                    return 

                    

class Node:
    def __init__(self, name, times = 1):
        self.name = name
        self.time = times
        self.preprocess()
        #self.neib = [] 

    def preprocess(self):
        self.name = self.name.lower() 



