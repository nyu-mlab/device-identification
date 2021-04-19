# device-identification

### Enviroment

**Python 3**

### Installation

```sh
git clone git@github.com:nyu-mlab/device-identification.git
cd device-identification
pip install -r requirements.txt
```

Download pretrained model from 

https://drive.google.com/file/d/1KIADDtIBZP18l3cIzIsktctfiAp_12WY/view?usp=sharing

And put it at ./data/model directory.

### Test

```python
#Open Python3 in device-identification
import api
api.get_vendor(oui: str, type_: "dns" or "port", data: list) -> str:
```

### Train

**Step I: Generate cluster based on editdistance.**

```shell
cd src
python3 cluster.py data_path dns_path port_path cluster_path rebuild 
```

**Arguments:**

[1]data_path: path to the main csv file(device_id, oui, vendor ...)

[2,3]dns_path & port_path: path to the corresponding dns/port csv file, pass a '' if none.

[4]cluster_path: path to save the cluster for the raw input.

[5]rebuild: True/False. Pass true if you have new data, otherwise pass False to read from cluster_path.



**Step II: Train models based on the cluster**

```shell
python3 train.py cluster_path feat_type method bayes_path LR_path
```

**Arguments:**

[1]cluster_path: path to the saved cluster.

[2]feat_type: dns/port.

[3]method: bayes/LR/mix.

[4,5]bayes_path LR_path: path to save the trained model, or load the models for mix model.





#### Meeting Logs

|    Date    |                            Event                             |
| :--------: | :----------------------------------------------------------: |
| 01/04/2021 | 1. Colab 2.Database(expected by 01/06) 3.Reminder:prepare, notes, recap |
| 01/25/2021 |   1.try unsupervised method 2.set up meeting every Monday    |
| 02/01/2021 |      1.try tfidf on extra infomation 2.try web crawler       |
| 03/05/2021 |      1.Build training and test set 2.new tf-idf method       |
| 03/19/2021 |           1.NLP model for dns 2. mix oui+dns model           |
| 03/29/2021 |                   1.coner cases 2.new data                   |
| 04/09/2021 |         1.tp-link 2.try original oui 3.code refactor         |
| 04/16/2021 |           1.code refactor 2.comments 2. Net_disco            |

#### Experiments

##### 1. Device_vendor

| Feature | Method  | Avg. Acc | std  |               Config                |
| :-----: | :-----: | :------: | :--: | :---------------------------------: |
|   OUI   |  Bayes  |   77%    | 0.3% |     Random 10% train, 10 times      |
|   DNS   | tf-idf  |   18%    |      |                                     |
|   DNS   | bow+NB  |   60%    |      |              80% train              |
|   DNS   | bow+LR  |   74%    |  1%  |              80% train              |
|   DNS   | bow+MLP |   69%    |      |              80% train              |
|   DNS   | bow+LR  |   68%    |      |              20% train              |
| DNS+OUI | Voting  |   80%    |      | all training data with both oui&dns |
| DNS+OUI | bow+LR  | **82%**  |      |         concat oui and dns          |

###### 1.1 On "espressif"

| Feature  | Method | Avg. Acc |      Config       |
| :------: | :----: | :------: | :---------------: |
| OUI+DNS  | Voting |   13%    | all training data |
|   DNS    |  Bow   |   37%    | all training data |
|   OUI    | Bayes  |    8%    | all training data |
| OUI+DNS  |  Bow   |   45%    |     123 data      |
| OUI+Port |  Bow   | **95%**  |     182 data      |

##### 2. Port Data(8359 in 16451), use oui to generate device vendor

| Feature  | Method | Avg. Acc | std  |   Config   |
| :------: | :----: | :------: | :--: | :--------: |
|   OUI    | Bayes  |   96%    | 0.5% | 10831 data |
| OUI+Port | bow+LR |   94%    |      | 6038 data  |
| OUI+Port | Voting | **96%**  |      | 5856 data  |

​	

