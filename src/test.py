# Taiyu Long@mLab
# 03/17/2021 
# main script to run model on test set



import pickle
import csv 
from dataset import *
from utils import *
from pprint import pprint
from tabulate import tabulate
from collections import defaultdict
import tldextract

#SAVE_PATH = '../data/test/raw_data.pickle'
SAVE_PATH = '../data/train/raw_data4port.pickle'

TFIDF_PATH = '../data/model/tf_idf'
LR_PATH = '../data/model/bow_lr'
#BAYES_PATH = '../data/model/bayes'
BAYES_PATH = '../data/model/bayes_oui_port'
PORT_LR_PATH = '../data/model/bow_lr_oui+port'

TEST_BAYES = '../results/oui+bayes.csv'
TEST_TFIDF = '../results/oui+tfidf.csv'
TEST_MIX = '../results/oui+mix.csv'



#BAYES_OUT = '../results/oui+bayes.out'

FIELDS = ['device_vendor', 'device_id', 'device_oui', 'dhcp_hostname' ,'netdisco_device_info', 'dns', 'port']

if __name__ == '__main__':

   with open(SAVE_PATH, 'rb') as fp:
        data = pickle.load(fp)
   with open(PORT_LR_PATH, 'rb') as fp:
        dns_data = pickle.load(fp)
   with open(BAYES_PATH, 'rb') as fp:
        oui_model = pickle.load(fp)

   dns_model,  dict_x, dict_y, = dns_data['model'],  dns_data['dx'], dns_data['dy']
   num2name = {v:k for k,v in dict_y.items()} 
   num2name[-1] = ''
   ret_bayes = [] ; ret_tfidf= []; ret_mix = []
   #cnt_bayes = cnt_tfidf = 0
   data_len = len(data)
   for i in range(data_len): 
        cur = data[i]
        device_id = cur[FIELDS[1]]
        expected = manual_rule(cur[FIELDS[0]].lower())
        oui = cur[FIELDS[2]][0]
        dns = cur[FIELDS[5]]

        # mix model
        featLen = len(dict_x)
        dns_one_hot = [0] * featLen
        oui_ret = [0, -1]; lr_ret = [0, -1] 
        for e in dns: 
            domain = tldextract.extract(e).domain
            if domain in dict_x: 
                dns_one_hot[dict_x[domain]] = 1
        #if len(oui) == 0 and len(dns) == 0: missing += 1; continue
        if oui in oui_model:
            name, prob = oui_model[oui][0]
            oui_ret = [prob, dict_y[name]] 
        prob = dns_model.predict_proba([dns_one_hot])[0]
        pred_class = dns_model.predict([dns_one_hot])[0]
        #print(prob, pred_class)
        if pred_class >= len(prob) or oui_ret[0] >= prob[pred_class]:
            inferred = oui_ret[1]
        else: inferred = pred_class
        inferred = num2name[inferred]
        #info = [device_id, expected, inferred,  is_equivalent(inferred, expected)] 
        info = [expected, inferred, oui , dns[:3] , is_equivalent(inferred, expected)] 
        ret_mix += [info]
        #print("processing {}-th data".format(i), end="\r") 
        #if i ==1000: break 
        ''' 
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
        # for oui
        if oui in bayes:
            inferred = bayes[oui][0][0]
        else:
            inferred = ''
        info = [device_id, expected, inferred, oui ,int(expected==inferred)] 
        ret_bayes += [info]
        '''

   #print(tabulate(ret_mix, headers= ["device_id","expected_vendor","inferred_vendor",  "is_same"]))
   print(tabulate(ret_mix, headers= ["expected_vendor","inferred_vendor","oui", "dns" ,"is_same"]))
   '''
   with open(TEST_BAYES, mode='w') as fp:
        csv_writer = csv.writer(fp, delimiter='/n', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer = csv.writer(fp) 
        csv_writer.writerow(["device_id","expected_vendor","inferred_vendor", "data_len", "is_same"])
        csv_writer.writerow(ret_bayes)

   with open(TEST_TFIDF, mode='w') as fp:
        csv_writer = csv.writer(fp, delimiter='/n', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer = csv.writer(fp) 
        csv_writer.writerow(["device_id","expected_vendor","inferred_vendor", "dns_number", "is_same"])
        csv_writer.writerow(ret_tfidf)
    '''
   with open(TEST_MIX, mode='w') as fp:
        csv_writer = csv.writer(fp, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        #csv_writer.writerow(["device_id","expected_vendor","inferred_vendor",  "is_same"])
        csv_writer.writerow(["expected_vendor","inferred_vendor", "oui", "dns" "is_same"])
        csv_writer.writerow(ret_mix)

   #dataObj = Dataset(graphObj.graph)
   #dataObj.train_test(FIELDS[5], 'tf_idf', 0.8)
