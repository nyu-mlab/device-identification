# Taiyu Long@mLab
# 01/29/2021 
# main script to run the cluster 



from cluster import * 
from dataset import *
import sys

DATA_PATH = '../data/train/device.csv'
DNS_PATH = '../data/train/dns.csv'
RAW_DATA = '../data/train/raw_data.pickle'
SAVE_PATH = '../data/train/cluster.pickle'
MODEL_PATH = '../data/model/'
FIELDS = ['device_vendor', 'device_id', 'device_oui', 'dhcp_hostname' ,'netdisco_device_info', 'dns']

if __name__ == '__main__':
   graphObj = Graph(DATA_PATH, DNS_PATH, SAVE_PATH, eval(sys.argv[1])) 
   raw_data = graphObj.read_data(DATA_PATH, DNS_PATH, '', True )

   with open(RAW_DATA, 'wb') as fp:
        pickle.dump(raw_data, fp)
   #print(raw_data)
   #dataObj = Dataset(graphObj.graph)

   if sys.argv[2] == 'tf_idf': 
        model = dataObj.train_test(FIELDS[5], 'tf_idf', 0.8)
        dataObj.save_model(model, MODEL_PATH+'tf_idf')
   if sys.argv[2] == 'bayes': 
        model = dataObj.train_test(FIELDS[2], 'bayes', 0.8)
        dataObj.save_model(model, MODEL_PATH+'bayes')


   '''
   tpb = set(dataObj.train_test(FIELDS[2], 'bayes', 0.8))
   tpt = set(dataObj.train_test(FIELDS[5], 'tf_idf', 0.8))
   diffb , difft = len(tpb - tpt), len(tpt - tpb)
   print(diffb / len(tpb), difft / len(tpt))
   '''

    
