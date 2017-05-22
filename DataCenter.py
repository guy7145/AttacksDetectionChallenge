import numpy as np
import csv

from keras.models import Model, Sequential
from keras.layers import Input, Dense

from ann import *


class DataCenter:
    labeled_processed_data_path = "Data/Processed_Data/labeled/"
    labels_file_path = "Data/Raw_Data/labels.csv";
    raw_data_dir_path = "Data/Raw_Data/FraudedRawData/";
    raw_data_filename = "User"
    raw_data_num_of_files = 40
    num_of_partitions = 150
    num_of_benign_sessions_per_user = 50

    compressed_sample_size = 5
    num_of_sample_per_partition = 100

    all_data = dict();

    raw_data = list()
    substitution = dict()
    vectorized_data = list()

    labeled_data_fraud = list()
    labeled_data_normal = list()
    labeled_data_unknown = list()

    labeled_ngrams_fraud = list()
    labeled_ngrams_normal = list()
    labeled_ngrams_unknown = list()
    all_ngrams = list()

    single_vector_compressed_size = 5
    encoder = 0

    def __init__(self):

        return

    @staticmethod
    def flatten_list(lst):
        flatten_list = list();
        for el in lst:
            flatten_list.extend(el)

        return flatten_list

    # def train_classifier(self):
    #
    #     x = []
    #     y = []
    #     x.extend(self.labeled_data_normal)
    #     y.extend(np.zeros(len(self.labeled_data_normal)))
    #     x.extend(self.labeled_data_fraud)
    #     y.extend(np.ones(len(self.labeled_data_fraud)))
    #
    #     x = np.array(x)
    #     y = np.array(y)
    #
    #     model = generate_classifier(x[0].shape)
    #     print("(training...)")
    #
    #     model.fit(x, y, epochs=300, batch_size=1, verbose=1, shuffle=True)
    #     return
    #
    # def train_encoder(self):
    #     xs = []
    #     ys = []
    #     for partition in self.labeled_data_normal:
    #         for sample in partition:
    #             xs.append(sample)
    #
    #     for partition in self.labeled_data_fraud:
    #         for sample in partition:
    #             xs.append(sample)
    #
    #     for partition in self.labeled_data_unknown:
    #         for sample in partition:
    #             xs.append(sample)
    #
    #     xs = np.array(xs)
    #
    #     (encoder, autoencoder) = generate_autoencoder(xs[0].shape, self.single_vector_compressed_size)
    #
    #     autoencoder.fit(xs, xs, epochs=0, batch_size=10, verbose=1, shuffle=True)
    #     self.encoder = encoder
    #
    #     return
    #
    # def compress_single_labeled_list(self, list_of_partitions):
    #     new_list = []
    #     for partition in list_of_partitions:
    #         # the compressed form of a partition is a 1d array the size of
    #         # <single_vector_compressed_size> * <num_of_sample_per_partition>
    #         compressed_partition = partition
    #         compressed_partition = self.encoder.predict(partition)
    #         # p = np.zeros(len(partition[0]))
    #         # for sample in partition:
    #         #    p += np.array(sample)
    #         flatten_partition = self.flatten_list(compressed_partition)
    #         new_list.append(flatten_partition)
    #     return new_list
    #
    # def compress_all_labeled_data(self):
    #
    #     self.labeled_data_fraud = self.compress_single_labeled_list(self.labeled_data_fraud)
    #     self.labeled_data_unknown = self.compress_single_labeled_list(self.labeled_data_unknown)
    #     self.labeled_data_normal = self.compress_single_labeled_list(self.labeled_data_normal)
    #
    #     return

    def user_to_user_sessions(self, rawdata):
        return (rawdata[i:i + self.num_of_sample_per_partition] for i in range(0, len(rawdata), self.num_of_sample_per_partition))

    def load_raw_data(self, i):
        raw_data = list()
        with open(self.raw_data_dir_path + self.raw_data_filename + str(i)) as datafile:
            lines = [line.rstrip('\n') for line in datafile.readlines()]
            raw_data.extend(lines)
        return list(self.user_to_user_sessions(raw_data))

    def load_all_raw_data(self):
        users_raw_data = list()
        for i in range(0, self.raw_data_num_of_files):
            users_raw_data.append(self.load_raw_data(i))
        return users_raw_data

    # def generate_substitution(self):
    #     word_set = set()
    #     for section in self.raw_data:
    #         for word in section:
    #             word_set.add(word)
    #     word_list = list(word_set)
    #     word_list.sort()
    #     for i in range(0, len(word_list)):
    #         self.substitution[word_list[i]] = i
    #
    #     return self.substitution

    # def vectorize_single(self, word, key_to_index):
    #     vec = np.zeros(len(self.substitution.keys()))
    #     vec[self.substitution[word]] = 1.0
    #     return vec
    #
    # def vectorize_all(self, data, key_to_index):
    #     for section in self.raw_data:
    #         processed_section = []
    #         for word in section:
    #             processed_section.append(self.vectorize_single(word))
    #         self.vectorized_data.append(processed_section)
    #
    #     return

    def generate_key_substitution(self, keys):
        sub = dict()
        for i in range(0, len(keys)):
            sub[keys[i]] = i
        return sub

    def vectorize_user_sessions(self, data, key_to_index):
        new_data = list()
        n = len(key_to_index.keys())
        for section in data:
            vec = np.zeros(n)
            for word in section:
                if word in key_to_index.keys():
                    vec[key_to_index[word]] += 1.0
            new_data.append(vec)
        return new_data

    def calculate_ngrams_from_wordlist(self, wordlist, n):
        all_ngrams = set()
        for i in range(0, len(wordlist) - n):
                ngram = ','.join(wordlist[i:i+n])
                all_ngrams.add(ngram)
        return list(all_ngrams)

    def user_sessions_to_ngrams(self, user_sessions, n):
        user_ngrams = list()
        for session in user_sessions:
            session_as_ngrams = self.calculate_ngrams_from_wordlist(session, n)
            for i in range(2, n):
                session_as_ngrams.extend(self.calculate_ngrams_from_wordlist(session, i))
            user_ngrams.append(session_as_ngrams)
        return user_ngrams

    # def generate_self_ngrams(self, n):
    #     self.labeled_ngrams_fraud = self.calculate_ngrams_from_features(self.labeled_data_fraud, n)
    #     self.labeled_ngrams_normal = self.calculate_ngrams_from_features(self.labeled_data_normal, n)
    #     self.labeled_ngrams_unknown = self.calculate_ngrams_from_features(self.labeled_data_unknown, n)
    #     return

    # def save_file(self, index, label, data, path):
    #     data_file = open(path + label + "/" + str(index), 'w')
    #     for vec in data:
    #         for vi in vec:
    #             data_file.write(str(vi))
    #             data_file.write(',')
    #         data_file.write('\n')
    #     data_file.close()
    #     return

    def attach_labels(self, all_users_sessions):
        all_users_labeled_sessions = list()

        with open(self.labels_file_path, "rt") as f:
            rows = enumerate(csv.reader(f, delimiter=","))

            for row, user in zip(rows, all_users_sessions):
                single_user_labeled_sessions = list()
                for label, session in zip(row[1], user):
                    if label == str(0):
                        label = 0
                    elif label == str(1):
                        label = 1
                    else:
                        label = 'unknown'

                    labeled_session = dict()
                    labeled_session['label'] = label
                    labeled_session['data'] = session

                    single_user_labeled_sessions.append(labeled_session)

                all_users_labeled_sessions.append(single_user_labeled_sessions)

        return all_users_labeled_sessions

    # def load_labels(self, data_to_label):
    #     with open(self.labels_file_path, "rt") as f:
    #         reader = csv.reader(f, delimiter=",")
    #         index = 0
    #         for row in enumerate(reader):
    #             for label in row[1]:
    #                 if label == str(0):
    #                     self.labeled_data_normal.append(np.array(data_to_label[index]))
    #                 elif label == str(1):
    #                     self.labeled_data_fraud.append(np.array(data_to_label[index]))
    #                 else:
    #                     self.labeled_data_unknown.append(np.array(data_to_label[index]))
    #                 index += 1
    #                 # print(str(index) + ":\t\t" + label)
    #     return
    #
    # # def save_labeled_data(self, path):
    # #     index = 0;
    # #
    # #     for sample in self.labeled_data_normal:
    # #         print(
    # #             "saving file " + str(index + 1) + " out of " + str(self.raw_data_num_of_files * self.num_of_partitions) + "(normal)")
    # #         self.save_file(index, "normal", sample, path)
    # #         index += 1
    # #
    # #     for sample in self.labeled_data_fraud:
    # #         print(
    # #             "saving file " + str(index + 1) + " out of " + str(self.raw_data_num_of_files * self.num_of_partitions) + "(fraud)")
    # #         self.save_file(index, "fraud", sample, path)
    # #         index += 1
    # #
    # #     for sample in self.labeled_data_unknown:
    # #         print(
    # #             "saving file " + str(index + 1) + " out of " + str(self.raw_data_num_of_files * self.num_of_partitions) + "(unknown)")
    # #         self.save_file(index, "unknown", sample, path)
    # #         index += 1
    # #
    # #     return

    def initialize(self, ngrams_size):
        # load data
        raw_data = self.load_all_raw_data()
        self.all_data['raw'] = raw_data

        # generate ngrams and substitutions
        all_users_sessions_ngrams = list()
        keys = list()
        for user in raw_data:
            user_sessions_as_ngrams = self.user_sessions_to_ngrams(user, ngrams_size)
            all_user_ngrams = list(set(self.flatten_list(user_sessions_as_ngrams[:self.num_of_benign_sessions_per_user])))
            all_users_sessions_ngrams.append(user_sessions_as_ngrams)
            keys.extend(all_user_ngrams)

        self.all_data['all_users_sessions_ngrams'] = all_users_sessions_ngrams
        self.all_data['all_users_substitutions'] = self.generate_key_substitution(list(set(keys)))


        all_users_session_ngrams_processed = list()
        substitution = self.all_data['all_users_substitutions']
        for user_sessions_as_ngrams in all_users_sessions_ngrams:
            all_users_session_ngrams_processed.append(self.vectorize_user_sessions(user_sessions_as_ngrams, substitution))

        self.all_data['all_users_session_ngrams_processed'] = all_users_session_ngrams_processed

        # load labeles
        labeled_data = self.attach_labels(all_users_session_ngrams_processed)
        self.all_data['labeled'] = labeled_data

        return
