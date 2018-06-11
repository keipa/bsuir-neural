from multiprocessing.dummy import Pool as ThreadPool
import numpy as np
import re
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer


train_data_set = 3300
test_data_set = 3500
# def getPrediction(train_data_set, test_data_set):
def getPrediction(pattern):

    all_data_set = test_data_set
    whole_data_set = 32818
    letter_dict = {
        "A": 1,
        "B": 2,
        "C": 3,
        "D": 4,
        "F": 5,
        "G": 6,
        "H": 7,
        "I": 8,
        "J": 9,
        "K": 10,
        "M": 11,
        "N": 12,
        "P": 13,
        "R": 14,
        "S": 15,
        "T": 16,
        "V": 17}

    data = pd.read_csv("SELECT_DISTINCT____d_DrugKey_____dr_cd_s_therapy_notnull.csv").as_matrix()
    clf = DecisionTreeClassifier()

    xtrainstr = np.array([])
    for chem in data[0:all_data_set, 1]: ## data limit
        xtrainstr = np.append(xtrainstr, re.sub("\ \|.*\|", "", chem))

    alldata = []
    for arr in CountVectorizer().fit_transform(xtrainstr).A:
        alldata.append(list(arr))

    castxtrain = alldata[0:train_data_set]
    castxtest = alldata[train_data_set:test_data_set]

    all_label = np.array([], dtype=object)
    for label in data[0:all_data_set, 5]:
        all_label = np.append(all_label, letter_dict[label])

    train_label = list(all_label)[0:train_data_set]
    actual_label = list(all_label)[train_data_set:test_data_set]

    clf.fit(castxtrain, train_label)

    p = clf.predict(castxtest)

    count = 0

    for i in range(test_data_set-train_data_set):
        count += 1 if p[i] == actual_label[i] else 0
    print(str(train_data_set) + ":" + str(test_data_set) + "=" +  str(float(count)/float(test_data_set-train_data_set)))



# urls = []
# # getPrediction(train_data_set, test_data_set)
# for i in range(3000,7000, 100):
#     for j in range(100, i, 200):
#         urls.append(str(j)+":"+str(i))
#         # getPrediction(j, i)
#
#
#
#
#
#
#
# # Make the Pool of workers
# pool = ThreadPool(8)
#
# # Open the urls in their own threads
# # and return the results
# results = pool.map(getPrediction, urls)
#
# #close the pool and wait for the work to finish
# pool.close()
# pool.join()
getPrediction("s")