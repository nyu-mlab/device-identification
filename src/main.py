# Taiyu Long@mLab
# 01/29/2021 
# main script to run the cluster 



from cluster import * 
from dataset import *
import sys
import numpy as np

DATA_PATH = '../data/train/device.csv'
DNS_PATH = '../data/train/dns.csv'
RAW_DATA = '../data/train/raw_data.pickle'
SAVE_PATH = '../data/train/cluster.pickle'
MODEL_PATH = '../data/model/'
FIELDS = ['device_vendor', 'device_id', 'device_oui', 'dhcp_hostname' ,'netdisco_device_info', 'dns']

if __name__ == '__main__':
   graphObj = Graph(DATA_PATH, DNS_PATH, SAVE_PATH, eval(sys.argv[1])) 
   graphObj.read_data(DATA_PATH, DNS_PATH, SAVE_PATH, eval(sys.argv[1]))

   #with open(RAW_DATA, 'wb') as fp:
   #     pickle.dump(raw_data, fp)
   #print(raw_data)
   dataObj = Dataset(graphObj.graph)
   ret = []
   if sys.argv[2] == 'bow': 
        #model, acc = dataObj.bow_softmax(FIELDS[5], 0.2)
        #model, acc = dataObj.train_test(FIELDS[5], 'tf_idf', 0.2)
        #for _ in range(10):
        #    ret += [dataObj.bow_softmax(FIELDS[5], 0.2)]
        #dataObj.save_model(model, MODEL_PATH+'tf_idf')
        #print(np.mean(ret), np.std(ret))
        #dataObj.save_model(model, MODEL_PATH+'bow_lr_dns+oui_test')

        dns_data = dataObj.load(MODEL_PATH + 'bow_lr_dns+oui_test_c=2') 
        oui_model = dataObj.load(MODEL_PATH + 'bayes')
        print(dataObj.mix_model(dns_data, oui_model)) 

   if sys.argv[2] == 'bayes': 
        for _ in range(10):
            model, acc = dataObj.train_test(FIELDS[2], 'bayes', 0.1)
            ret += [acc]
        print(np.mean(ret), np.std(ret))
        dataObj.save_model(model, MODEL_PATH+'bayes')


   '''
   tpb = set(dataObj.train_test(FIELDS[2], 'bayes', 0.8))
   tpt = set(dataObj.train_test(FIELDS[5], 'tf_idf', 0.8))
   diffb , difft = len(tpb - tpt), len(tpt - tpb)
   print(diffb / len(tpb), difft / len(tpt))
   '''

    
