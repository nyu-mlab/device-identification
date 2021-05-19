# Taiyu Long@mLab
# 01/29/2021 
# main script to run the cluster 



from cluster import * 
from dataset import *
import sys
import numpy as np



if __name__ == '__main__':


   graphObj = Graph(save_path = sys.argv[1]) # build the graph
   dataObj = Dataset(graphObj.graph) # build dataset for training based on cluster
   ret = []

   if sys.argv[3] == 'LR': 
        model, acc = dataObj.bow_softmax(sys.argv[2] , 0.2)
        dataObj.save_model(model, sys.argv[4])

   if sys.argv[3] == 'bayes': 
        # since bayes model is very fast, we train it for 10 times and see the average performance
        for _ in range(10):
            model, acc, stat = dataObj.train_test(sys.argv[2], sys.argv[3]  , 0.2)
            ret += [acc]
        mean , std = np.mean(ret), np.std(ret)
        print("Avg Acc: {}, Std: {}".format(mean, std))
        #dataObj.draw_graph(stat) # you can draw class stats here
        #print(tp)
        dataObj.save_model(model,  sys.argv[4] )

   if sys.argv[3] == 'mix': 
        # voting method
        data_model = dataObj.load(sys.argv[4])
        feat_model = dataObj.load(sys.argv[5])
        acc, oui, feat = dataObj.mix_model(data_model, feat_model)
        print("Acc: {}, using {}: {}, using {}: {}".format(acc, sys.argv[4], oui,sys.argv[5], feat))

