import pandas as pd
from collections import Counter
from editdistance import distance

device_df = pd.read_csv('../data/device.csv').fillna('')
fields = device_df.columns[1:5]
#print(fields)
#print("data:", len(device_df))
#for i in range(4):
#    print(fields[i], len(Counter(device_df[fields[i]])))
i = 2
#print(fields[i], len(Counter(device_df[fields[i]])))
for j in range(len(device_df[fields[i]])):
    device_df[fields[i]][j] = device_df[fields[i]][j].lower()
#print(fields[i], len(Counter(device_df[fields[i]])))
#print(fields[i], Counter(device_df[fields[i]]))

cnt = Counter(device_df[fields[i]])
print(fields[i], len(cnt))
change = {}
thre = 2
standard = 10
temp = []
for k, v in cnt.items():
    if v >= standard: 
        temp += [k]    

for k, v in list(cnt.items()):
    if v >= standard: continue
    for s in temp:
        if distance(s, k) <= thre: 
            change[k] = (s, v)
            cnt[s] += cnt[k]
            del cnt[k]
            break

#print(fields[i], len(cnt))
print(fields[i], cnt)
#print(change)




