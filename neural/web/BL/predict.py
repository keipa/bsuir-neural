import numpy as np
import re
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer





def predict(inputPattern):
    train_data_set = 3300
    test_data_set = 3301

    all_data_set = 3300

    therapy_dict ={
    1: "Alimentary Tract and Metabolic Products",
    2: "Blood and Clotting Products",
    3: "Cardiovascular Products",
    4: "Dermatological Products",
    5: "Formulations",
    6: "Genitourinary Products( and Sex Hormones)",
    7: "Hormones(excluding Sex Hormones)",
    8: "Immunological",
    9: "Anti - infectives(Systemic)",
    10: "Anticancer Products",
    11: "Musculoskeletal Products",
    12: "Neurological Products",
    13: "Antiparasitic Products",
    14: "Respiratory Products",
    15: "Sensory Products",
    16: "Biotechnology",
    17: "Other Products"
    }

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
        "V": 17
    }

    data = pd.read_csv("G:/Labs/bsuir-neural/neural/web/BL/SELECT_DISTINCT____d_DrugKey_____dr_cd_s_therapy_notnull.csv").as_matrix()
    clf = DecisionTreeClassifier()
    xtrainstr = np.array([])
    for chem in data[0:all_data_set, 1]: ## data limit
        xtrainstr = np.append(xtrainstr, re.sub("\ \|.*\|", "", chem))
    xtrainstr = np.append(xtrainstr,  re.sub("\ \|.*\|", "", inputPattern))
    alldata = []
    for arr in CountVectorizer().fit_transform(xtrainstr).A:
        alldata.append(list(arr))
    castxtrain = alldata[0:train_data_set-1]
    castxtest = [alldata[len(alldata)-1]]
    all_label = np.array([], dtype=object)
    for label in data[0:all_data_set, 5]:
        all_label = np.append(all_label, letter_dict[label])
    train_label = list(all_label)[0:train_data_set-1]
    clf.fit(castxtrain, train_label)
    p = clf.predict(castxtest)
    return therapy_dict[p[0]]
