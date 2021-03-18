# Taiyu Long@mLab
# 03/17/2021 
# main script to run model on test set



import pickle
from dataset import *
from pprint import pprint
from tabulate import tabulate
from collections import defaultdict

DATA_PATH = '../data/test/device.csv'
DNS_PATH = '../data/test/dns.csv'
SAVE_PATH = '../data/test/raw_data.pickle'
TFIDF_PATH = '../data/model/tf_idf'
BAYES_PATH = '../data/model/bayes'

BAYES_OUT = '../results/oui+bayes.out'

FIELDS = ['device_vendor', 'device_id', 'device_oui', 'dhcp_hostname' ,'netdisco_device_info', 'dns']

if __name__ == '__main__':

   with open(SAVE_PATH, 'rb') as fp:
        data = pickle.load(fp)
   with open(TFIDF_PATH, 'rb') as fp:
        tfidf = pickle.load(fp)
   with open(BAYES_PATH, 'rb') as fp:
        bayes = pickle.load(fp)
    
   ret_bayes = [] ; ret_tfidf= []  
   #cnt_bayes = cnt_tfidf = 0
   data_len = len(data)
   for i in range(data_len): 
        cur = data[i]
        device_id = cur[FIELDS[1]]
        expected = cur[FIELDS[0]].lower()
        oui = cur[FIELDS[2]][0]
        dns = cur[FIELDS[5]]

        #for dns
        inferred = defaultdict(int)
        inferred[''] = 0
        for e in dns:
            for oui, f in tfidf[e]:
                #print(oui,f)
                inferred[oui] += f
        inferred = sorted(inferred.items(), key=lambda x: -x[1])[0][0]        
        info = [device_id, expected, inferred, len(dns), int(expected==inferred)] 
        ret_tfidf += [info] 
        '''
        if oui in bayes:
            inferred = bayes[oui][0][0]
        else:
            inferred = ''
        info = [device_id, expected, inferred, oui ,int(expected==inferred)] 
        ret_bayes += [info]
        '''
   print(tabulate(ret_tfidf, headers= ["device_id","expected_vendor","inferred_vendor", "dns_number", "is_correct"]))


   #dataObj = Dataset(graphObj.graph)
   #dataObj.train_test(FIELDS[5], 'tf_idf', 0.8)
