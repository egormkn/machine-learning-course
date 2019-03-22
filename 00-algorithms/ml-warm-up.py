# Copyright 2018 Artem Ivanov
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.feature_selection import SelectKBest, RFE, RFECV
from sklearn.model_selection import train_test_split, RepeatedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

np.random.seed(239)

train = pd.read_csv("../input/ml1-data/train.csv", index_col=0)
train_data = train.values[:, :27]
train_target = train.values[:, -1]

exch = dict(zip(['A', 'B', 'C', 'D', 'E', 'F', 'G'], range(7)))
for i in range(0, train_data.shape[0]):
    for j in range(20, train_data.shape[1]):
        train_data[i, j] = exch[train_data[i, j]]

train_data, test_data, train_target, test_target = train_test_split(train_data, train_target, test_size=0.2)

total_k_neigh = -1
total_k_select = -1
total_acc = -1
pair = -1
total_pair = -1

f = open("log.txt", "w")

# RFECV + SVC
print("RFECV + SVC")
svc = SVC(kernel="linear")
selector = RFECV(estimator=svc, cv=10).fit(train_data, train_target)
classifier = SVC()
good = 0
total = 0
k_neigh = 0
k_select = 0
acc = 0
rkf = RepeatedKFold(n_splits=10, n_repeats=5)
for train_ind, check_ind in rkf.split(train_data):
    data_train, target_train, data_check, target_check = \
        train_data[train_ind], train_target[train_ind], train_data[check_ind], train_target[check_ind]

    classifier.fit(selector.transform(data_train), target_train)

    ans = classifier.predict(selector.transform(data_check))
    good += sum([ans[i] == target_check[i] for i in range(len(ans))])
    total += len(ans)
if good / total > acc:
    pair = 3
    acc = good / total
    k_select = selector.n_features_
    # print("k_select", k_select, "good", good, "total", total, "per", good/total, "pair", pair)

trans = RFE(estimator=svc, n_features_to_select=k_select).fit(train_data, train_target)
ans = SVC().fit(trans.transform(train_data), train_target).predict(trans.transform(test_data))
good = sum([ans[i] == test_target[i] for i in range(len(ans))])
total = len(ans)
print("RFECV + SVC", "good =", good, "total =", total, "accuracy =", good / total * 100, "%", file = f)
if good / total > total_acc:
    total_acc = good / total
    total_pair = 3
    total_k_neigh = k_neigh
    total_k_select = k_select

# RFECV + KNN
print("RFECV + KNN")
svc = SVC(kernel="linear")
selector = RFECV(estimator=svc, cv=10).fit(train_data, train_target)
k_neigh = 0
k_select = 0
acc = 0
for k in range(1, 20):
    classifier = KNeighborsClassifier(n_neighbors=k)
    good = 0
    total = 0
    rkf = RepeatedKFold(n_splits=10, n_repeats=5)
    for train_ind, check_ind in rkf.split(train_data):
        data_train, target_train, data_check, target_check = \
            train_data[train_ind], train_target[train_ind], train_data[check_ind], train_target[check_ind]

        classifier.fit(selector.transform(data_train), target_train)

        ans = classifier.predict(selector.transform(data_check))
        good += sum([ans[i] == target_check[i] for i in range(len(ans))])
        total += len(ans)
    if good / total > acc:
        pair = 1
        acc = good / total
        k_neigh = k
        k_select = selector.n_features_
        # print("k_neigh", k_neigh, "k_select", k_select, "good", good, "total", total, "per", good/total, "pair", pair)

trans = RFE(estimator=svc, n_features_to_select=k_select).fit(train_data, train_target)
ans = KNeighborsClassifier(n_neighbors=k_neigh).fit(trans.transform(train_data), train_target).predict(
    trans.transform(test_data))
good = sum([ans[i] == test_target[i] for i in range(len(ans))])
total = len(ans)
print("RFECV + KNN", "good =", good, "total =", total, "accuracy =", good / total * 100, "%", file = f)
if good / total > total_acc:
    total_acc = good / total
    total_pair = 1
    total_k_neigh = k_neigh
    total_k_select = k_select

# SelectKBest + SVC
print("SelectKBest + SVC")
k_neigh = 0
k_select = 0
acc = 0
for t in range(1, train_data.shape[1] + 1):
    selector = SelectKBest(k=t)
    classifier = SVC()
    good = 0
    total = 0
    rkf = RepeatedKFold(n_splits=10, n_repeats=5)
    for train_ind, check_ind in rkf.split(train_data):
        data_train, target_train, data_check, target_check = \
            train_data[train_ind], train_target[train_ind], train_data[check_ind], train_target[check_ind]

        selector.fit(data_train, target_train)
        classifier.fit(selector.transform(data_train), target_train)

        ans = classifier.predict(selector.transform(data_check))
        good += sum([ans[i] == target_check[i] for i in range(len(ans))])
        total += len(ans)
    if good / total > acc:
        acc = good / total
        k_select = t
        pair = 2
        # print("k_select", k_select, "good", good, "total", total, "per", good/total, "pair", pair)

trans = SelectKBest(k=k_select).fit(train_data, train_target)
ans = SVC().fit(trans.transform(train_data), train_target).predict(trans.transform(test_data))
good = sum([ans[i] == test_target[i] for i in range(len(ans))])
total = len(ans)
print("SelectKBest + SVC", "good =", good, "total =", total, "accuracy =", good / total * 100, "%", file = f)
if good / total > total_acc:
    total_acc = good / total
    total_pair = 2
    total_k_neigh = k_neigh
    total_k_select = k_select

# SelectKBest + KNN
print("SelectKBest + KNN")
k_neigh = 0
k_select = 0
acc = 0
for t in range(1, train_data.shape[1] + 1):
    selector = SelectKBest(k=t)
    for k in range(1, 20):
        classifier = KNeighborsClassifier(n_neighbors=k)
        good = 0
        total = 0
        rkf = RepeatedKFold(n_splits=10, n_repeats=5)
        for train_ind, check_ind in rkf.split(train_data):
            data_train, target_train, data_check, target_check = \
                train_data[train_ind], train_target[train_ind], train_data[check_ind], train_target[check_ind]

            selector.fit(data_train, target_train)
            classifier.fit(selector.transform(data_train), target_train)

            ans = classifier.predict(selector.transform(data_check))
            good += sum([ans[i] == target_check[i] for i in range(len(ans))])
            total += len(ans)
        if good / total > acc:
            pair = 0
            acc = good / total
            k_neigh = k
            k_select = t
            # print("k_neigh", k_neigh, "k_select", k_select, "good", good, "total", total, "per", good/total, "pair", pair)

trans = SelectKBest(k=k_select).fit(train_data, train_target)
ans = KNeighborsClassifier(n_neighbors=k_neigh).fit(trans.transform(train_data), train_target).predict(
    trans.transform(test_data))
good = sum([ans[i] == test_target[i] for i in range(len(ans))])
total = len(ans)
print("SelectKBest + KNN", "good =", good, "total =", total, "accuracy =", good / total * 100, "%", file = f)
if good / total > total_acc:
    total_acc = good / total
    total_pair = 0
    total_k_neigh = k_neigh
    total_k_select = k_select

# OneHotEncoder
from sklearn.preprocessing import OneHotEncoder

train = pd.read_csv("../input/ml1-data/train.csv", index_col=0)
train_data = train.values[:, :27]
train_target = train.values[:, -1]

exch = dict(zip(['A', 'B', 'C', 'D', 'E', 'F', 'G'], range(7)))
for i in range(0, train_data.shape[0]):
    for j in range(20, train_data.shape[1]):
        train_data[i, j] = exch[train_data[i, j]]

ohe = OneHotEncoder(sparse=False, categorical_features=range(20, 27))
train_data = ohe.fit_transform(train_data)

train_data, test_data, train_target, test_target = train_test_split(train_data, train_target, test_size=0.2)

# RFECV + SVC
print("RFECV + SVC + OneHotEncoder")
svc = SVC(kernel="linear")
selector = RFECV(estimator=svc, cv=10).fit(train_data, train_target)
classifier = SVC()
good = 0
total = 0
k_neigh = 0
k_select = 0
acc = 0
rkf = RepeatedKFold(n_splits=10, n_repeats=5)
for train_ind, check_ind in rkf.split(train_data):
    data_train, target_train, data_check, target_check = \
        train_data[train_ind], train_target[train_ind], train_data[check_ind], train_target[check_ind]

    classifier.fit(selector.transform(data_train), target_train)

    ans = classifier.predict(selector.transform(data_check))
    good += sum([ans[i] == target_check[i] for i in range(len(ans))])
    total += len(ans)
if good / total > acc:
    pair = 7
    acc = good / total
    k_select = selector.n_features_
    # print("k_select", k_select, "good", good, "total", total, "per", good/total, "pair", pair)

trans = RFE(estimator=svc, n_features_to_select=k_select).fit(train_data, train_target)
ans = SVC().fit(trans.transform(train_data), train_target).predict(trans.transform(test_data))
good = sum([ans[i] == test_target[i] for i in range(len(ans))])
total = len(ans)
print("RFECV + SVC + OneHotEncoder", "good =", good, "total =", total, "accuracy =", good / total * 100, "%", file = f)
if good / total > total_acc:
    total_acc = good / total
    total_pair = 7
    total_k_neigh = k_neigh
    total_k_select = k_select

# RFECV + KNN
print("RFECV + KNN + OneHotEncoder")
svc = SVC(kernel="linear")
selector = RFECV(estimator=svc, cv=10).fit(train_data, train_target)
k_neigh = 0
k_select = 0
acc = 0
for k in range(1, 20):
    classifier = KNeighborsClassifier(n_neighbors=k)
    good = 0
    total = 0
    rkf = RepeatedKFold(n_splits=10, n_repeats=5)
    for train_ind, check_ind in rkf.split(train_data):
        data_train, target_train, data_check, target_check = \
            train_data[train_ind], train_target[train_ind], train_data[check_ind], train_target[check_ind]

        classifier.fit(selector.transform(data_train), target_train)

        ans = classifier.predict(selector.transform(data_check))
        good += sum([ans[i] == target_check[i] for i in range(len(ans))])
        total += len(ans)
    if good / total > acc:
        pair = 5
        acc = good / total
        k_neigh = k
        k_select = selector.n_features_
        # print("k_neigh", k_neigh, "k_select", k_select, "good", good, "total", total, "per", good/total, "pair", pair)

trans = RFE(estimator=svc, n_features_to_select=k_select).fit(train_data, train_target)
ans = KNeighborsClassifier(n_neighbors=k_neigh).fit(trans.transform(train_data), train_target).predict(
    trans.transform(test_data))
good = sum([ans[i] == test_target[i] for i in range(len(ans))])
total = len(ans)
print("RFECV + KNN + OneHotEncoder", "good =", good, "total =", total, "accuracy =", good / total * 100, "%", file = f)
if good / total > total_acc:
    total_acc = good / total
    total_pair = 5
    total_k_neigh = k_neigh
    total_k_select = k_select

# SelectKBest + SVC
print("SelectKBest + SVC + OneHotEncoder")
k_neigh = 0
k_select = 0
acc = 0
for t in range(1, train_data.shape[1] + 1):
    selector = SelectKBest(k=t)
    classifier = SVC()
    good = 0
    total = 0
    rkf = RepeatedKFold(n_splits=10, n_repeats=5)
    for train_ind, check_ind in rkf.split(train_data):
        data_train, target_train, data_check, target_check = \
            train_data[train_ind], train_target[train_ind], train_data[check_ind], train_target[check_ind]

        selector.fit(data_train, target_train)
        classifier.fit(selector.transform(data_train), target_train)

        ans = classifier.predict(selector.transform(data_check))
        good += sum([ans[i] == target_check[i] for i in range(len(ans))])
        total += len(ans)
    if good / total > acc:
        acc = good / total
        k_select = t
        pair = 6
        # print("k_select", k_select, "good", good, "total", total, "per", good/total, "pair", pair)

trans = SelectKBest(k=k_select).fit(train_data, train_target)
ans = SVC().fit(trans.transform(train_data), train_target).predict(trans.transform(test_data))
good = sum([ans[i] == test_target[i] for i in range(len(ans))])
total = len(ans)
print("SelectKBest + SVC + OneHotEncoder", "good =", good, "total =", total, "accuracy =", good / total * 100, "%", file
      = f)
if good / total > total_acc:
    total_acc = good / total
    total_pair = 6
    total_k_neigh = k_neigh
    total_k_select = k_select

# SelectKBest + KNN
print("SelectKBest + KNN + OneHotEncoder")
k_neigh = 0
k_select = 0
acc = 0
for t in range(1, train_data.shape[1] + 1):
    selector = SelectKBest(k=t)
    for k in range(1, 20):
        classifier = KNeighborsClassifier(n_neighbors=k)
        good = 0
        total = 0
        rkf = RepeatedKFold(n_splits=10, n_repeats=5)
        for train_ind, check_ind in rkf.split(train_data):
            data_train, target_train, data_check, target_check = \
                train_data[train_ind], train_target[train_ind], train_data[check_ind], train_target[check_ind]

            selector.fit(data_train, target_train)
            classifier.fit(selector.transform(data_train), target_train)

            ans = classifier.predict(selector.transform(data_check))
            good += sum([ans[i] == target_check[i] for i in range(len(ans))])
            total += len(ans)
        if good / total > acc:
            pair = 4
            acc = good / total
            k_neigh = k
            k_select = t
            # print("k_neigh", k_neigh, "k_select", k_select, "good", good, "total", total, "per", good/total, "pair", pair)

trans = SelectKBest(k=k_select).fit(train_data, train_target)
ans = KNeighborsClassifier(n_neighbors=k_neigh).fit(trans.transform(train_data), train_target).predict(
    trans.transform(test_data))
good = sum([ans[i] == test_target[i] for i in range(len(ans))])
total = len(ans)
print("SelectKBest + KNN + OneHotEncoder", "good =", good, "total =", total, "accuracy =", good / total * 100, "%", file
      = f)
if good / total > total_acc:
    total_acc = good / total
    total_pair = 4
    total_k_neigh = k_neigh
    total_k_select = k_select

# result
"""
if total_pair == 0:
    trans = SelectKBest(k=total_k_select).fit(train_data, train_target)
    ans = KNeighborsClassifier(n_neighbors=total_k_neigh).fit(trans.transform(train_data), train_target).predict(trans.transform(test_data))
elif total_pair == 1:
    trans = RFE(estimator=svc, n_features_to_select=total_k_select).fit(train_data, train_target)
    ans = KNeighborsClassifier(n_neighbors=total_k_neigh).fit(trans.transform(train_data), train_target).predict(trans.transform(test_data))
elif total_pair == 2:
    trans = SelectKBest(k=total_k_select).fit(train_data, train_target)
    ans = SVC().fit(trans.transform(train_data), train_target).predict(trans.transform(test_data))
elif total_pair == 3:
    trans = RFE(estimator=svc, n_features_to_select=total_k_select).fit(train_data, train_target)
    ans = SVC().fit(trans.transform(train_data), train_target).predict(trans.transform(test_data))

good = sum([ans[i] == test_target[i] for i in range(len(ans))])
total = len(ans)
print("Total", good, total, good / total)
"""

# predictions

predict = pd.read_csv("../input/ml1-data/test.csv", index_col=0)
predict_data = predict.values

for i in range(0, predict_data.shape[0]):
    for j in range(20, predict_data.shape[1]):
        predict_data[i, j] = exch[predict_data[i, j]]

if pair in range(4, 8):
    predict_data = ohe.fit_transform(predict_data)

if total_pair in [0, 4]:
    trans = SelectKBest(k=total_k_select).fit(train_data, train_target)
    ans = KNeighborsClassifier(n_neighbors=total_k_neigh).fit(trans.transform(train_data), train_target).predict(
        trans.transform(predict_data))
elif total_pair in [1, 5]:
    trans = RFE(estimator=svc, n_features_to_select=total_k_select).fit(train_data, train_target)
    ans = KNeighborsClassifier(n_neighbors=total_k_neigh).fit(trans.transform(train_data), train_target).predict(
        trans.transform(predict_data))
elif total_pair in [2, 6]:
    trans = SelectKBest(k=total_k_select).fit(train_data, train_target)
    ans = SVC().fit(trans.transform(train_data), train_target).predict(trans.transform(predict_data))
elif total_pair in [3, 7]:
    trans = RFE(estimator=svc, n_features_to_select=total_k_select).fit(train_data, train_target)
    ans = SVC().fit(trans.transform(train_data), train_target).predict(trans.transform(predict_data))

pd.DataFrame(data=ans, index=range(1, 6634, 2), columns=['class']).to_csv("answer.csv", header=True, index_label='id')
f.close()
