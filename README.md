# device-identification

#### Meeting Logs

|    Date    |                            Event                             |
| :--------: | :----------------------------------------------------------: |
| 01/04/2021 | 1. Colab 2.Database(expected by 01/06) 3.Reminder:prepare, notes, recap |
| 01/25/2021 |   1.try unsupervised method 2.set up meeting every Monday    |
| 02/01/2021 |      1.try tfidf on extra infomation 2.try web crawler       |
| 03/05/2021 |      1.Build training and test set 2.new tf-idf method       |
| 03/19/2021 |           1.NLP model for dns 2. mix oui+dns model           |
| 03/29/2021 |                   1.coner cases 2.new data                   |

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
|   OUI    | Bayes  |   97%    | 0.2% | 10382 data |
| OUI+Port | bow+LR |   95%    |      | 5802 data  |
| OUI+Port | Voting | **97%**  |      | 5802 data  |

​	

