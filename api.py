# Taiyu Long@mLab
# 04/12/2021 

# model path
BAYES_DNS = './data/model/bayes'
LR_DNS = './data/model/dns_all'
LR_PORT = './data/model/port_all'

import pickle
import oui_parser
import re


oui_regex = re.compile(r'[0-9a-fA-F]{6}')


def get_vendor(oui: str, dns: list, port: list, disco: dict) -> str:
    """
    Note that `oui` could be a vendor's name (as defined by IEEE) or the
    6-character OUI code (basically the first 6 characters of the MAC address.)

    Universal api, provide list of str or [] for each argument.
    Note that you can only provide either dns or port, which means at least one of them should be empty. 
    """

    oui_model = load(BAYES_DNS) 
    if port: data_dict = load(LR_PORT) 
    else: data_dict = load(LR_DNS)  #if disco alone, use this

    # OUI could be a string of [0-9a-f]{6} or in English. 
    if is_oui_code(oui):
        oui = oui.lower()
        # Look up the vendor
        vendor = oui_parser.get_vendor(oui)
        if vendor != '':
            oui = vendor 

    # decode info for lr model
    feat = dns + port + disco_proc(disco)
    data_model,  dict_x, dict_y, = data_dict['model'],  data_dict['dx'], data_dict['dy']
    num2name = {v:k for k,v in dict_y.items()}
    num2name[-1] = '' 
    one_hot = [0] * len(dict_x)
    for e in feat: 
        if e != '' and e in dict_x: one_hot[dict_x[e]] = 1
    
    oui_ret = lr_ret = None  
    if oui and feat: # get voting result
       oui_ret = [0, -1]; lr_ret = [0, -1] 
       if oui in oui_model:
           name, prob = oui_model[oui][0]
           oui_ret = [prob, dict_y[name]] 
       prob = data_model.predict_proba([one_hot])[0]
       pred_class = data_model.predict([one_hot])[0]
       if pred_class >= len(prob) or oui_ret[0] >= prob[pred_class]:
           inferred = oui_ret[1]
       else: inferred = pred_class
       return num2name[inferred]

    if oui: return oui_model[oui][0][0]

    return num2name[data_model.predict([one_hot])[0]]


def disco_proc(disco_dict):
    '''same as the preprocessing in cluster
    '''
    if not disco_dict: return []
    disco = []
    filt = ['device','local', 'homekit', 'basement', 'bedroom']
    for info in disco_dict.values():
        if type(info) == dict: disco += self.disco(info)
        elif type(info) == str: 
            cur = re.split('[^a-zA-Z]', info)
            for e in cur:
                e = e.lower()
                if len(e) > 4 and e not in filt: disco += [e]

        return list(set(disco))

def load(load_path):
    with open(load_path, 'rb') as fp:
        return pickle.load(fp)


def is_oui_code(oui_string):

    match = oui_regex.match(oui_string)
    return match is not None
    

def test():        
    print(is_oui_code('aabbcc'))
    print(is_oui_code('AAbbcc'))
    print(is_oui_code('Intel Corporate'))

    print(get_vendor('Intel Corporate' , [] ,  ['8081'], []))
    print(get_vendor('780cb8' , [], ['8081'], []))
    print(get_vendor('Intel Corporate' , [] , [] , []))
    print(get_vendor('' , [], ['8081'], []))

    # The following tests don't work yet @TODO(Taiyu) resolved@Danny
    print(get_vendor('NETGEAR' , ['netgear'], [], []))
    print(get_vendor('NETGEAR' , [], [], []))
    print(get_vendor('' , ['netgear'], [], []))

    # test for disco
    disco = {'ssdp_description': 'http://192.168.1.122:80/dms/device.xml', 'name': 'HDHomeRun DMS 13142937', 'upnp_device_type': 'urn:schemas-upnp-org:device:MediaServer:1', 'model_number': 'HDHR3-CC', 'model_name': 'HDHomeRun PRIME', 'host': '192.168.1.122', 'device_type': 'dlna_dms', 'serial': '13142937', 'udn': 'uuid:527F67D9-33EF-3B9F-A8A2-D934DE3DFAC4', 'port': 80, 'manufacturer': 'Silicondust'}

    print(get_vendor('' , [], [], disco))
    print(get_vendor('Silicondust Engineering Ltd' , [], [], disco))
    print(get_vendor('Silicondust Engineering Ltd' , [], [], []))



if __name__ == '__main__':
    test()
