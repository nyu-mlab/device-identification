# Taiyu Long@mLab
# 01/29/2021 
# main script to run the cluster 



from cluster import * 
from dataset import *
import sys

DATA_PATH = '../data/device.csv'
DNS_PATH = '../data/dns.csv'
SAVE_PATH = '../data/cluster.pickle'
FIELDS = ['device_vendor', 'device_id', 'device_oui', 'dhcp_hostname' ,'netdisco_device_info', 'dns']

if __name__ == '__main__':
   graphObj = Graph(DATA_PATH, DNS_PATH, SAVE_PATH, eval(sys.argv[1])) 
   dataObj = Dataset(graphObj.graph)
   dataObj.train_test(FIELDS[2], 'bayes', 0.8)

    
