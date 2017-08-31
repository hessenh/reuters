import numpy as np
import pandas as pd
from bs4 import BeautifulSoup, SoupStrainer
from keras.preprocessing.text import text_to_word_sequence
import os
from gensim.models.word2vec import Word2Vec

class Case(object):
    def __init__(self, case):
        self.topics = []
        self.title = []
        self.body = []
        self.__split_type = 'lewissplit'
        self.set_type = 'NOT-USED'
        self.__extract_fields(case)

    def __extract_fields(self, case):
        self.__extract_text(case)
        self.__extract_topics(case)
        self.__extract_lewis(case)

    def __extract_topics(self, case):
        topics = case.findAll('d')
        for topic in topics:
            self.topics.append(topic.text)

    def __extract_text(self, case):
        text = case.find('text')
        title = text.title
        if title != None:
            self.title = self.__text_to_word(title.text)
        body = text.body
        if body != None:
            self.body = self.__text_to_word(body.text)

    def __extract_lewis(self, case):
        self.set_type = case[self.__split_type]

    def __text_to_word(self, text):
        text = text.encode('ascii', 'ignore')
        return text_to_word_sequence(
            text,
            filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
            lower=True,
            split=" ")

class Dataset(object):
    def __init__(self, path, should_load_word2vec, number_of_features, number_of_words):
        self.__cases = []
        self.__docs = []
        self.__topics = []
        self.__word2vec_model_name = 'model.word2vec'

        self.X_train = []
        self.X_test = []
        self.y_train = []
        self.y_test = []

        self.__all_categories = []
        self.__word2vec_model = None
        self.__number_of_features = number_of_features
        self.__max_number_of_words = number_of_words
        self.__number_of_train = 0
        self.__number_of_test = 0

        self.__load_topics(path)
        self.__load_cases(path)

        if should_load_word2vec:
            self.__load_word2vec_model(path)
        else:
            self.__save_word2vec_model(path)

        self.__populate_data_set()

    def __load_cases(self, path):
        files = os.listdir(path)
        sgm_files = [i for i in files if i.endswith('.sgm')]
        for i in range(0, len(sgm_files)):
            print 'Loading', sgm_files[i]
            d = open(path + sgm_files[i], 'r')
            text = d.read()
            d.close()
            soup = BeautifulSoup(text, 'html.parser')
            contents = soup.findAll('reuters')
            for i in range(0, len(contents)):
                # Extract case from sgm-file
                case = Case(contents[i])
                if case.set_type != 'NOT-USED':
                    self.__cases.append(case)
                    self.__docs.append(case.body)
                    self.__topics.append(case.topics)

                if case.set_type == 'TRAIN':
                    self.__number_of_train += 1
                elif case.set_type == 'TEST':
                    self.__number_of_test += 1
        print 'Number of train samples', self.__number_of_train
        print 'Number of test samples', self.__number_of_test

    def __load_topics(self, path):
        self.__all_categories = pd.read_csv(path + 'all-topics-strings.lc.txt', header=None).values


    def __load_word2vec_model(self, path):
        self.__word2vec_model = Word2Vec.load(path + self.__word2vec_model_name)

    def __save_word2vec_model(self, path):
        self.__word2vec_model = Word2Vec(
            self.__docs,
            size=self.__number_of_features,
            min_count=1,
            window=10)
        self.__word2vec_model.save(path + self.__word2vec_model_name)
        print 'Model saved as', self.__word2vec_model_name

    def __populate_data_set(self):
        # Create placeholders for data and labels
        self.X_train = np.zeros((self.__number_of_train, self.__max_number_of_words, self.__number_of_features))
        self.y_train = np.zeros((self.__number_of_train, len(self.__all_categories)))
        self.X_test = np.zeros((self.__number_of_test, self.__max_number_of_words, self.__number_of_features))
        self.y_test = np.zeros((self.__number_of_test, len(self.__all_categories)))

        train_index = 0
        test_index = 0

        for i in range(0, len(self.__docs)):
            # Populate train and test data with word-2-vec embedded docs
            for j in range(0, len(self.__docs[i])):
                word = self.__docs[i][j]
                if j == self.__max_number_of_words:
                    break
                elif word in self.__word2vec_model:
                    if self.__cases[i].set_type == 'TRAIN':
                        self.X_train[train_index, j, :] = self.__word2vec_model[word]
                    else:
                        self.X_test[test_index, j, :] = self.__word2vec_model[word]

            # Populate train and test labels with an encoded vector
            # e.g [1, 0, 0, 1]
            label = np.zeros(len(self.__all_categories))
            for j in range(0, len(self.__all_categories)):
                if self.__all_categories[j] in self.__topics[i]:
                    label[j] = 1.0
            if self.__cases[i].set_type == 'TRAIN':
                self.y_train[train_index, :] = label
                train_index += 1
            else:
                self.y_test[test_index, :] = label
                test_index += 1


    def get_train_test(self):
        return self.X_train, self.X_test, self.y_train, self.y_test
