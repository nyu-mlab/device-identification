# device-identification

#### Meeting Logs

|    Date    |                            Event                             |
| :--------: | :----------------------------------------------------------: |
| 01/04/2021 | 1. Colab 2.Database(expected by 01/06) 3.Reminder:prepare, notes, recap |
| 01/25/2021 |   1.try unsupervised method 2.set up meeting every Monday    |
| 02/01/2021 |      1.try tfidf on extra infomation 2.try web crawler       |
| 03/05/2021 |      1.Build training and test set 2.new tf-idf method       |

#### Experiments

##### 1. Device_vendor

| Feature | Method | Avg. Acc | std  |           Config           |
| :-----: | :----: | :------: | :--: | :------------------------: |
|   OUI   | Bayes  |   77%    | 0.3% | Random 10% train, 10 times |
|   DNS   | tf-idf |   18%    |      |                            |

