# Taiyu Long@mLab
# 01/29/2021 
# main script to run the cluster 



from cluster import * 
from dataset import *
import sys
import numpy as np
import pandas as pd

'''
DATA_PATH = '../data/train/device.csv'
DNS_PATH = '../data/train/dns.csv'
RAW_DATA = '../data/train/raw_data4port.pickle'
SAVE_PATH = '../data/train/cluster.pickle'
#SAVE_PATH = '../data/train/cluster4port.pickle'
MODEL_PATH = '../data/model/'
PORT_DEVICE_DATA = '../data/train/devices4port.csv'

PORT_DATA = '../data/train/syn_scan_ports.3df6bc62.csv'
'''


if __name__ == '__main__':

   #graphObj = Graph(DATA_PATH, DNS_PATH,None , SAVE_PATH, eval(sys.argv[1])) 
   #graphObj.read_data(DATA_PATH, DNS_PATH,None ,SAVE_PATH, eval(sys.argv[1]))

   graphObj = Graph(save_path = sys.argv[1])
   dataObj = Dataset(graphObj.graph)

   #with open(RAW_DATA, 'wb') as fp:
   #     pickle.dump(raw_data, fp)
   #print(raw_data)
   ret = []
   if sys.argv[3] == 'LR': 
        model, acc = dataObj.bow_softmax(sys.argv[2] , 0.1)
        #for _ in range(10):
        #    ret += [dataObj.bow_softmax(FIELDS[5], 0.2)[0]]
        #dataObj.save_model(model, MODEL_PATH+'tf_idf')
        #print(np.mean(ret), np.std(ret))
        dataObj.save_model(model, sys.argv[4])


   if sys.argv[3] == 'bayes': 
        for _ in range(10):
            model, acc = dataObj.train_test(sys.argv[2], sys.argv[3]  , 0.8)
            ret += [acc]
        print(np.mean(ret), np.std(ret))
        dataObj.save_model(model,  sys.argv[4] )

   if sys.argv[3] == 'mix': 
        data_model = dataObj.load(sys.argv[4])
        feat_model = dataObj.load(sys.argv[5])
        acc, oui, feat = dataObj.mix_model(data_model, feat_model)
        print("Acc: {}, using {}: {}, using {}: {}".format(acc, sys.argv[4], oui,sys.argv[5], feat))

        


    
         


   '''
   tpb = set(dataObj.train_test(FIELDS[2], 'bayes', 0.8))
   tpt = set(dataObj.train_test(FIELDS[5], 'tf_idf', 0.8))
   diffb , difft = len(tpb - tpt), len(tpt - tpb)
   print(diffb / len(tpb), difft / len(tpt))
   '''

    
