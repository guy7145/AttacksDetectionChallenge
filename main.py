import random

import math

import copy
from DataCenter import *

import numpy as np

from sklearn.svm import OneClassSVM

from AttacksDetectionChallenge.DataCenter import DataCenter
from AttacksDetectionChallenge.ann import generate_classifier

training_verbosity = 2
negative_example_clones = 1
num_of_anns = 40
ngram_size = 3
batch_size = 2
epochs_number = 5
hidden_layer_to_input_size_ratio = 0.002

_evaluate = False
_predict = True
_prediction_classifiers_late_training = True
num_of_tested_users = 10
total_number_of_users = 40

encoder_training_start_printout_prefix = "---------------------------------------------------"


def evaluate_classifier(classifier, xs, ys):

    tp = tn = fp = fn = 0
    predicted = classifier.predict(xs)

    actual_pred = obtain_actual_predictions(predicted)
    negatives = positives = 0
    # print(predicted)
    for i in range(len(xs)):
        if ys[i] == 0 and actual_pred[i] == 0:
            tn += 1
            negatives += 1
        elif ys[i] == 0 and actual_pred[i] == 1:
            fp += 1
            negatives += 1
        elif ys[i] == 1 and actual_pred[i] == 0:
            fn += 1
            positives += 1
        elif ys[i] == 1 and actual_pred[i] == 1:
            tp += 1
            positives += 1

    print("tp: {}; fn: {}; tn: {}; fp: {}".format(tp, fn, tn, fp))
    print("score: {}".format(9*tp + tn))
    print("test data size: {} positives; {} negatives; total: {}".format(positives, negatives, positives + negatives))

    return 9*tp + tn


def obtain_actual_predictions(predicted):
    actual_pred = list()
    for p in predicted:
        if p > 0.5:
            actual_pred.append(1)
        else:
            actual_pred.append(0)
    return actual_pred


def main():
    print("initializing dc...")
    dc = DataCenter()
    dc.initialize(ngrams_size=ngram_size)

    normal_sessions = dc.all_data['raw'][:dc.num_of_benign_sessions_per_user]
    all_users_ngrams = dc.all_data['all_users_session_ngrams_processed']

    training_set = list()
    for user in all_users_ngrams:
        training_set.append(user[:50])

    # num_of_anns = len(all_users_ngrams)
    print("generating anns... ({})".format(num_of_anns))
    nok = len(dc.all_data['all_users_substitutions'].keys())
    prototype = generate_classifier(num_of_keys=nok, ratio=hidden_layer_to_input_size_ratio)
    anns = list()
    for i in range(num_of_anns):
        # print("input_img size = {}".format(nok))
        print("generating {} out of {}".format(i + 1, num_of_anns))
        ann = copy.deepcopy(prototype)
        anns.append(ann)
    print("done.")

    print("training anns... ({})".format(num_of_anns))
    i = 0;
    for user_ngrams, ann in zip(training_set, anns):
        train_ann(ann, i, training_set, user_ngrams, not _prediction_classifiers_late_training)
        i += 1

    if _evaluate:
        print("evaluating... ({})".format(num_of_tested_users))
        average_score = 0;
        i = 0
        for i, ann in zip(range(num_of_tested_users), anns):
            labeled_sessions = dc.all_data['labeled'][i]
            xs = [ls['data'] for ls in labeled_sessions][50:]
            ys = [ls['label'] for ls in labeled_sessions][50:]
            average_score += evaluate_classifier(ann, np.array(xs), np.array(ys))
            i += 1
        average_score = average_score/i
        print("average score: {}".format(average_score))

    if _predict:
        for i in range(num_of_tested_users, 40):
            if _prediction_classifiers_late_training:
                train_ann(anns[i], i, training_set, training_set[i], _prediction_classifiers_late_training)
            labeled_sessions = dc.all_data['labeled'][i]
            xs = [ls['data'] for ls in labeled_sessions][50:]
            predictions = obtain_actual_predictions(anns[i].predict(np.array(xs)))
            print(",".join([str(p) for p in predictions]))

    return


def train_ann(ann, i, training_set, user_ngrams, _train_prediction_classifier):
    print("{} {}/{}".format(encoder_training_start_printout_prefix, i + 1, num_of_anns))
    if (i < num_of_tested_users and _evaluate) or (i >= num_of_tested_users and _train_prediction_classifier):
        x_train, y_train = list(), list()
        for user_sessions, j in zip(training_set, range(len(training_set))):
            if j != i:
                x_train.extend(user_sessions)
        random.shuffle(x_train)
        y_train.extend(np.ones(len(x_train)))

        # creating a training set with natural label distributions
        # x_train = x_train[:9 * len(user_ngrams)]

        normal_examples = list()
        for k in range(negative_example_clones):
            normal_examples.extend(user_ngrams)
        random.shuffle(normal_examples)

        x_train.extend(normal_examples)
        y_train.extend(np.zeros(len(normal_examples)))

        ann.fit(np.array(x_train), np.array(y_train), epochs=epochs_number, batch_size=batch_size,
                verbose=training_verbosity, shuffle=True)


# def main_svm():
#     print("initializing dc...")
#     dc = DataCenter()
#     dc.initialize(ngrams_size=ngram_size)
#
#     normal_sessions = dc.all_data['raw'][:dc.num_of_benign_sessions_per_user]
#     all_users_ngrams = dc.all_data['all_users_session_ngrams_processed']
#
#     training_set = list()
#     for user in all_users_ngrams:
#         training_set.append(user[:50])
#
#     # num_of_autoencoders = len(all_users_ngrams)
#     num_of_autoencoders = 40
#     print("generating auto encoders... ({})".format(num_of_autoencoders))
#     SVMs = list()
#     for i in range(num_of_autoencoders):
#         nok = len(dc.all_data['all_users_substitutions'][i].keys())
#         print("input_img size = {}".format(nok))
#         svm = OneClassSVM(kernel='poly', degree=3, cache_size=1000, tol=0.05, shrinking=False, nu=0.4)
#         SVMs.append(svm)
#
#     print("training auto encoders... ({})".format(num_of_autoencoders))
#     i = 1;
#     for user_ngrams, svm in zip(training_set, SVMs):
#         print("{} {}/{}".format(encoder_training_start_printout_prefix, i, num_of_autoencoders))
#         i += 1
#         user_ngrams = np.array(user_ngrams)
#         svm.fit(user_ngrams)
#
#     num_of_tested_users = 10
#     print("\nevaluating... ({})".format(num_of_tested_users))
#     average_score = 0
#     for i in range(num_of_tested_users):
#         labeled_sessions = dc.all_data['labeled'][i]
#         xs = [ls['data'] for ls in labeled_sessions][50:]
#         ys = [ls['label'] for ls in labeled_sessions][50:]
#         average_score += evaluate_classifier(SVMs[i], xs, ys)
#     average_score = average_score / num_of_tested_users
#
#     print("average score: {}".format(average_score))
#
#     for i in range(num_of_tested_users, 40):
#         labeled_sessions = dc.all_data['labeled'][i]
#         xs = [ls['data'] for ls in labeled_sessions][50:]
#         predictions = SVMs[i].predict(xs)
#         predictions = [int((p+1)/2) for p in predictions]
#         print(",".join([str(p) for p in predictions]))
#
#     return

if __name__ == "__main__":
    main()
