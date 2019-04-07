import json
import csv
import os
import pandas as pd
import numpy as np
import glob
import math
import my_main
import xlsxwriter
import copy
from gensim import utils
from my_utils import print_dict_to_file
from my_utils import print_dicts_to_file
from my_utils import linear_regression
from my_utils import drop_constant_columns
from pre_processing import text_clean

# path to the FakeNewsNet dataset
base_path = "data/FakeNewsNet/"


def read_fakenews_dataset():
    """
    reading the converted dataset
    """
    try:
        with open(base_path + 'fakenewsnet.csv') as f:
            reader = csv.reader(f)
            next(reader)
            data = [r for r in reader]
        return data
    except:
        print("fakenewsnet.csv does not exist.")


def user_spread_count(source):
    """
    counting the number of times each user spreads news
    :param source: the source of articles in FakeNewsNet which can be either PolitiFact or Buzzfeed
    :return:
    """
    with open(base_path + source + "/" + source + "NewsUser.txt", 'r') as file:
        user_spread = {}
        for line in file:
            tmp = line.replace('\n', '').split('\t')
            if tmp[1] in user_spread:
                user_spread[tmp[1]] += int(tmp[2])
            else:
                user_spread[tmp[1]] = int(tmp[2])
        return user_spread


def count_followers(source):
    """
    counting the number of followers of each user
    :param source: the source of articles in FakeNewsNet which can be either PolitiFact or Buzzfeed
    :return:
    """
    followers = {}
    with open(base_path + source + "/" + source + "UserUser.txt", 'r') as file:
        current_user = -1
        for line in file:
            tmp = line.replace('\n', '').split('\t')
            # it means that it is a new user
            if current_user != tmp[1]:
                current_user = tmp[1]
                followers[tmp[1]] = 1
            # otherwise, increment the number of followers for current user
            else:
                followers[tmp[1]] += 1
        return followers


def news_spread_count(source):
    """
    counting the number of spread of all the news documents and number of users who spread them
    :param source: the source of articles in FakeNewsNet which can be either PolitiFact or Buzzfeed
    :return:
    """
    with open(base_path + source + "/" + source + "NewsUser.txt", 'r') as file:
        news_spread = {}
        news_spread_distinct = {}

        for line in file:
            tmp = line.replace('\n', '').split('\t')
            news_id = tmp[0]
            # for the number of times a news is spread
            if news_id in news_spread:
                news_spread[news_id] += int(tmp[2])
            else:
                news_spread[news_id] = int(tmp[2])

            # for counting the number of distinct users who spread news
            if news_id in news_spread_distinct:
                news_spread_distinct[news_id] += 1
            else:
                news_spread_distinct[news_id] = 1

        return news_spread, news_spread_distinct


def get_news_ids():
    """
    :return: ids of all the news documents in the dataset
    """
    buzzfeed_news_ids = {}
    politifact_news_ids = {}
    i = 1

    with open(base_path + "/BuzzFeed/News.txt", 'r') as file:
        for line in file:
            buzzfeed_news_ids[line.replace('\n', '')] = i
            i += 1
    i = 1
    with open(base_path + "/PolitiFact/News.txt", 'r') as file:
        for line in file:
            politifact_news_ids[line.replace('\n', '')] = i
            i += 1
    return buzzfeed_news_ids, politifact_news_ids


def reformat_dataset():
    """
    reformatting the original fakenewsnet data to a clean csv file.
    :return: one CSV file containing all the information from the dataset + a text file of all the text content of documents
    we create the text file to be used in future for topic modeling.
    """

    # creating a list of docs for topic modelling purposes
    all_docs = []
    all_docs_gensim = []

    all_files = os.listdir(base_path + "all")
    all_files = sorted(all_files)
    fake_spread_all = 0
    real_spread_all = 0
    fake_affected_all = 0
    real_affected_all = 0
    global_id = 1

    politi_news_spread, politi_news_spread_distinct = news_spread_count("PolitiFact")
    buzz_news_spread, buzz_news_spread_distinct = news_spread_count("BuzzFeed")
    buzzfeed_news_ids, politifact_news_ids = get_news_ids()
    # statistics
    stat = {"polifake": 0, "polireal": 0, "buzzfake": 0, "buzzreal": 0}
    with open(base_path + "processed/fakenewsnet.csv", 'w+') as csv_file:
        f_writer = csv.writer(csv_file, delimiter=',')
        f_writer.writerow(["Id", "OriginalId", "Title", "Body", "FactChecker", "Source", "Link", "Spread_count", "Spread_count_distinct", "Label"])
        for file in all_files:
            if file.startswith("Buzz") or file.startswith("Politi"):
                with open(base_path + "all/" + file, 'r') as current_file:
                    page_source = json.load(current_file)
                    # getting original id
                    tmp = file.split('-')
                    fact_checker = tmp[0].split('_')[0]
                    news_label = tmp[0].split('_')[1]

                    if "PolitiFact" in tmp[0]:
                        news_id = str(politifact_news_ids[tmp[0]])
                        spread_count = politi_news_spread[news_id]
                        distinct_user_spread = politi_news_spread_distinct[news_id]
                    elif "BuzzFeed" in tmp[0]:
                        news_id = str(buzzfeed_news_ids[tmp[0]])
                        spread_count = buzz_news_spread[news_id]
                        distinct_user_spread = buzz_news_spread_distinct[news_id]
                    try:
                        news_title = text_clean(page_source["title"], True, True, False, 1)
                        news_body = text_clean(page_source["text"], True, True, False, 1)
                        all_docs.append(news_title)
                        all_docs.append(news_body)

                        # list of cleaned docs for topic modelling purposes
                        all_docs_gensim.append(' '.join(utils.simple_preprocess(news_body)))
                        all_docs_gensim.append(' '.join(utils.simple_preprocess(news_title)))

                        f_writer.writerow([global_id,
                                           tmp[0],
                                           news_title,
                                           news_body,
                                           fact_checker,
                                           page_source["source"] if "source" in page_source else "",
                                           page_source["url"] if "url" in page_source else "",
                                           spread_count,
                                           distinct_user_spread,
                                           news_label.lower()
                                           ])
                        global_id += 1
                        # updating statistics
                        if fact_checker.startswith("Poli"):
                            if news_label == "Fake":
                                stat["polifake"] += 1
                            else:
                                stat["polireal"] += 1
                        else:
                            if news_label == "Fake":
                                stat["buzzfake"] += 1
                            else:
                                stat["buzzreal"] += 1
                        if news_label == "Fake":
                            fake_spread_all += spread_count
                        else:
                            real_spread_all += spread_count

                    except Exception as e:
                        print(e)

        # writing all docs into a text file to further use in topic model training
        with open(base_path + 'processed/fakenewsnet_gensim.txt', 'w') as text_file:
            for doc in all_docs:
                text_file.write(doc + '\n')

        # writing all docs into a text file to further use in topic model training
        with open(base_path + 'processed/fakenewsnet.txt', 'w') as text_file:
            for doc in all_docs:
                text_file.write(doc + '\n')

        print("Fake PolitiFact: " + str(stat["polifake"]))
        print("Real PolitiFact: " + str(stat["polireal"]))
        print("Sum PolitiFact: " + str(stat["polifake"] + stat["polireal"]))
        print("---------------------")
        print("Fake Buzzfeed: " + str(stat["buzzfake"]))
        print("Real Buzzfeed: " + str(stat["buzzreal"]))
        print("Sum Buzzfeed: " + str(stat["buzzfake"] + stat["buzzreal"]))
        print("---------------------")
        print("Sum all: " + str(stat["polifake"] + stat["polireal"] + stat["buzzfake"] + stat["buzzreal"]))
        print("Fake spread count: " + str(fake_spread_all))
        print("Real spread count: " + str(real_spread_all))
        print("---------------------")
        print("Fake affected count: " + str(fake_affected_all))
        print("Real affected count: " + str(real_affected_all))


def create_prerequisites():
    """
    creating the files needed for reformatting the FakeNewsNet dataset
    :return:
    """
    followers_dict = count_followers("PolitiFact")
    print_dict_to_file(base_path + "PolitiFact/PolitiFactUserFollowers.txt", followers_dict)

    followers_dict = count_followers("BuzzFeed")
    print_dict_to_file(base_path + "BuzzFeed/BuzzFeedUserFollowers.txt", followers_dict)

    news_spread, news_spread_distinct = news_spread_count("PolitiFact")
    print_dicts_to_file(base_path + "PolitiFact/PolitiFactNewsSpread.txt", news_spread_distinct, news_spread)

    news_spread, news_spread_distinct = news_spread_count("BuzzFeed")
    print_dicts_to_file(base_path + "BuzzFeed/BuzzFeedNewsSpread.txt", news_spread_distinct, news_spread)

    user_spread = user_spread_count("PolitiFact")
    print_dict_to_file(base_path + "PolitiFact/PolitiFactUserSpread.txt", user_spread)

    user_spread = user_spread_count("BuzzFeed")
    print_dict_to_file(base_path + "BuzzFeed/BuzzFeedUserSpread.txt", user_spread)

    print("Prerequisites are created successfully.")


def create_fakenewsnet_text_cohmetrix():
    """
    this method creates text files of title and body of news articles
    separately to be used as the input of coh-metrix
    :return:
    """
    # all_data = pd.read_csv(base_path + "fakenewsnet.csv")
    # creating a text file of all the records
    print("[-->] Start creating text file...")
    for file in glob.glob(base_path + "all/*.json"):
        with open(file, 'r') as current_file:
            try:
                page_source = json.load(current_file)
                # getting original id
                tmp = file.split('/')
                original_id = tmp[len(tmp)-1].split('-')[0]
                news_title = my_main.text_clean(page_source["title"], True, True, False, 1)
                news_body = my_main.text_clean(page_source["text"], True, True, False, 1)
                with open(base_path + 'cohmetrix/title/' + original_id + '.txt', 'w+') as text_file:
                    text_file.write(news_title + '\n')
                    text_file.close()
                with open(base_path + 'cohmetrix/body/' + original_id + '.txt', 'w+') as text_file:
                    text_file.write(news_body + '\n')
                    text_file.close()
            except Exception as e:
                print(e)
    print("[-->] Text file is created successfully")


def coh_regresssion_before_pca():
    main_data = pd.read_csv(base_path + "fakenewsnet.csv")
    coh_data = pd.read_csv(base_path + "cohmetrix/cohout/fake_cohmetrix.csv")
    x_columns = list(coh_data.iloc[:, 1:])
    # y_lin = main_data.iloc[:, 7]
    x_reg = []
    y_reg = []
    x_lin = []
    y_lin = []
    fact_labels = []
    for item in coh_data.iterrows():
        x = item[1][1:]
        tmp = item[1][0].split('\\')
        my_id = tmp[len(tmp)-1].replace('.txt', '')
        my_record = main_data.loc[main_data['OriginalId'] == my_id]
        truth_label = my_record.values[0][11]
        y = my_record.values[0][7]
        if y != 0:
            x_lin.append(x)
            y_lin.append(y)
            # adding fact checking label (fake/real)
            if truth_label == "fake":
                fact_labels.append(1)
            else:
                fact_labels.append(0)

    y_lin = np.array(y_lin)
    # define vectorized sigmoid
    log_func = np.vectorize(math.log)
    y_lin = log_func(y_lin)

    # converting list to dataframe
    x_lin = pd.DataFrame(np.array(x_lin).reshape(len(x_lin), len(x_columns)), columns=x_columns)
    # dropping constant value columns
    # x_lin = np.array(x_lin)
    x_lin = drop_constant_columns(x_lin)

    # scaling the data
    # scaler = MinMaxScaler(feature_range=(0, 1))
    # x_lin = scaler.fit_transform(x_lin)

    causal_info = pd.read_excel(base_path + "fakenewsnet_causal.xlsx")

    x_full = copy.deepcopy(x_lin)
    x_full["CREL"] = causal_info["rels"]
    x_full["CRELWD"] = causal_info["rels/words"]
    x_full["CRELSC"] = causal_info["rels/sentences"]
    x_full["shares"] = y_lin
    x_full["label"] = fact_labels
    x_full = x_full.fillna(0)
    writer = pd.ExcelWriter(base_path + 'faker.xlsx')
    x_full.to_excel(writer, 'Sheet1')
    writer.save()

    print(" === Experiment 1 ===")
    linear_regression(x_lin, y_lin, "")

    # print(" === Experiment 2 ===")
    # x_rel = causal_info.iloc[:, len(list(causal_info))-3:len(list(causal_info))]
    # x_rel = x_rel.fillna(0)
    # linear_regression(x_rel, y_lin, "")


def coh_regression_after_pca():
    # loading disney shares files
    all_data = pd.read_csv(base_path + 'r_coh_fake.csv')
    model_features = list(all_data)

    x_lin = all_data.iloc[:, 1:len(model_features) - 1]
    y_lin = all_data.iloc[:, len(model_features) - 1]

    # scaling the data
    model_features = list(x_lin)

    print(" === Experiment 1 ===")
    linear_regression(x_lin, y_lin, model_features)


# step 0: creating prerequisites files
# create_prerequisites()

# step 1: reformatting the data
reformat_dataset()

# step 2: creating the inputs for coh-metrix
# create_fakenewsnet_text_cohmetrix()

# experiments
# 1: running LSA model and using Foltz method
# run_lsa_fakenewsnet_foltz()

# 2: running and loading the LDA model
# run_lda_fakenewsnet_topic_coherence(10)

# 3: running GloVe on fakenewsnet
# run_glove_fakenewsnet(0)
# run_glove_mongotwitter(1)

# 4: to extract causal relations of news documents
# fakenewsnet_causal_relations()
