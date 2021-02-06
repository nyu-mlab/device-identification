# Taiyu Long@mLab
# 01/29/2021 
# main script to run the cluster 



from cluster import * 
import sys

DATA_PATH = '../data/device.csv'
DNS_PATH = '../data/dns.csv'
SAVE_PATH = '../data/cluster.pickle'
if __name__ == '__main__':
   cluster = Graph(DATA_PATH, DNS_PATH, SAVE_PATH, eval(sys.argv[1])) 

    
