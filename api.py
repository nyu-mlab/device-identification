# Taiyu Long@mLab
# 04/12/2021 

# model path
BAYES_DNS = './data/model/bayes'
BAYES_PORT = './data/model/bayes_oui_port'
LR_DNS = './data/model/bow_lr_oui+dns'
LR_PORT = './data/model/bow_lr_oui+port'

import pickle

def get_vendor(oui: str, type_: str, data: list) -> str:
    if type_ == 'dns': 
        oui_model = load(BAYES_DNS) 
        data_dict = load(LR_DNS)
    elif type_ == 'port':
        oui_model = load(BAYES_PORT) 
        data_dict = load(LR_PORT)
    else: raise NotImplementedError
    data_model,  dict_x, dict_y, = data_dict['model'],  data_dict['dx'], data_dict['dy']
    num2name = {v:k for k,v in dict_y.items()}
    num2name[-1] = '' 
    one_hot = [0] * len(dict_x)
    for e in data+[oui]: 
        if e in dict_x: one_hot[dict_x[e]] = 1

    if oui and data:
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

    if oui:
       return oui_model[oui][0][0]
    if data:
       return num2name[data_model.predict([one_hot])[0]]
    return 'unknown'


def load(load_path):
    with open(load_path, 'rb') as fp:
        return pickle.load(fp)


def test():        

    print(get_vendor('Intel Corporate' , 'port', ['8081']))
    print(get_vendor('Intel Corporate' , 'port', []))
    print(get_vendor('' , 'port', ['8081']))

    # The following tests don't work yet @TODO(Taiyu)
    print(get_vendor('NETGEAR' , 'dns',   ['netgear']))
    print(get_vendor('NETGEAR' , 'dns',   []))
    print(get_vendor('' , 'dns',   ['netgear']))


if __name__ == '__main__':
    test()
