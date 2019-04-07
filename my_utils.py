import math
import xlrd
import pickle
import re
import scipy.spatial.distance as distance
import numpy as np
import pandas as pd
from itertools import islice
from gensim.models.wrappers import LdaMallet
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from bs4 import BeautifulSoup
import urllib3
import operator
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def print_dict_to_file(file_path, my_dict):
    with open(file_path, 'a') as file:
        for key, value in my_dict.items():
            file.write(key + "\t" + str(value) + "\n")


def print_dicts_to_file(file_path, dict_1, dict_2):
    # dicts share the same key
    with open(file_path, 'a') as file:
        for key, value in dict_1.items():
            file.write(key + "\t" + str(value) + "\t" + str(dict_2[key]) + "\n")


def save_file(file_name, file_content):
    with open(file_name, 'wb') as handle:
        pickle.dump(file_content, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_file(file_name):
    with open(file_name, 'rb') as handle:
        file_content = pickle.load(handle)
    return file_content


def save_pickle_file(file_name, file_content):
    with open('data/pickle/' + file_name + '.pickle', 'wb') as handle:
        pickle.dump(file_content, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle_file(file_name):
    with open('data/pickle/' + file_name + '.pickle', 'rb') as handle:
        file_content = pickle.load(handle)
    return file_content


def save_gensim_file(file, name):
    file.save('data/gensim_models/' + name)


def load_gensim_file(file_name):
    return LdaMallet.load('data/gensim_models/' + file_name)


def cosine_similarity(vector_1, vector_2):
    return 1 - distance.cosine(vector_1, vector_2)


def read_excel_data(excel_file_path, sheet_index):
    ''' this method is used for reading all rows of an excel file '''
    all_rows = []
    xl_workbook = xlrd.open_workbook(excel_file_path)
    xl_sheet = xl_workbook.sheet_by_index(sheet_index)
    # reading all the docs in which we want to find causal relations
    for row_idx in range(1, xl_sheet.nrows):
        all_rows.append(xl_sheet.row(row_idx))
        # if we want to read a single cell
        # xl_sheet.cell(row_idx, 0).value
    return all_rows


def html_image_count(url):
    ''' this method returns the number of images included in a html '''
    http = urllib3.PoolManager()
    r = http.request('GET', url)
    soup = BeautifulSoup(r.data)
    pat_1 = re.compile('<img alt=')
    img_1 = pat_1.findall(str(soup.body))
    pat_2 = re.compile('<img src=')
    img_2 = pat_2.findall(str(soup.body))
    return len(img_1) + len(img_2)


def logistic_regression(x, y, model_features):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    logreg = LogisticRegression(class_weight="balanced", solver='lbfgs')
    logreg.fit(x_train, y_train)
    print("\n\nLogistic regression report")
    print("=========================")
    if model_features != "":
        # print(logreg.coef_)
        feature_coeff = list(zip(logreg.coef_[0], model_features))
        for item in sorted(feature_coeff, key=lambda x: abs(x[0])):
            print(item)
    y_pred = logreg.predict(x_test)
    print('Accuracy: {:.2f}'.format(logreg.score(x_test, y_test)))
    print(classification_report(y_test, y_pred))
    print("Confusion matrix")
    print(confusion_matrix(y_test, y_pred))


def linear_regression(x, y, model_features):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    linreg = LinearRegression()
    linreg.fit(x_train, y_train)
    print("\n\nLinear regression report")
    print("=========================")
    if model_features != "":
        feature_coeff = list(zip(linreg.coef_, model_features))
        # print(linreg.coef_)
        for item in sorted(feature_coeff, key=lambda x: abs(x[0])):
            print(item)
    y_pred = linreg.predict(x_test)
    # coefficients
    # print('Coefficients: \n', linreg.coef_)
    # mean squared error: the lower the better
    print("\nMean Squared Error (MSE): %.2f" % mean_squared_error(y_test, y_pred))
    # # R-squared: the closer to 1 (%100) the better
    print('R^2: %.2f' % r2_score(y_test, y_pred))
    print("Intercept: " + str(linreg.intercept_))

    # another way of printing coefficients with their names
    # coeff_df = pd.DataFrame(linreg.coef_, x.columns, columns=['Coefficient'])
    # print(coeff_df)


def model_test_cross(x, y, predictor):
    cv_results = cross_validate(predictor, x, y, cv=5, return_train_score = True)
    print(cv_results)


def varimax_rotation(Phi, gamma=1, q=20, tol=1e-6):
    from numpy import eye, asarray, dot, sum, diag
    from numpy.linalg import svd
    p, k = Phi.shape
    R = eye(k)
    d = 0
    for i in range(q):
        d_old = d
        Lambda = dot(Phi, R)
        u, s, vh = svd(dot(Phi.T, asarray(Lambda)**3 - (gamma/p) * dot(Lambda, diag(diag(dot(Lambda.T, Lambda))))))
        R = dot(u, vh)
        d = sum(s)
        if d/d_old < tol:
            break
    return dot(Phi, R)


def list_dictionary_top_n(dict, n):
    return list(islice(dict.items(), n))


def drop_constant_columns(df):
    nunique = df.apply(pd.Series.nunique)
    cols_to_drop = nunique[nunique == 1].index
    return df.drop(cols_to_drop, axis=1)
