import heapq
import random

import math

import copy
from DataCenter import *

import numpy as np
from keras.models import load_model
from keras.models import save_model

from sklearn.svm import OneClassSVM

from AttacksDetectionChallenge.DataCenter import DataCenter
from AttacksDetectionChallenge.ann import generate_classifier

greatestn_n = 30
num_of_eval_steps = 500
# num_of_eval_steps = 1
eval_step_size = 0.001
eval_epsilon = 0.001
training_verbosity = 0
evaluation_verbosity = 0
negative_example_clones = 7
num_of_anns = 40
ngram_size = 2
batch_size = 1
epochs_number = 3
hidden_layer_to_input_size_ratio = 0.002

_save_anns = False
_load_anns = False
_train_anns = not _load_anns
_evaluate = True
_predict = True
_prediction_classifiers_late_training = False
num_of_tested_users = 10
total_number_of_users = 40
_model_save_path = "saved_models\\ann_"

encoder_training_start_printout_prefix = "---------------------------------------------------"


def evaluate_classifier(classifier, xs, ys, epsilon, n):

    tp = tn = fp = fn = 0
    predicted = classifier.predict(xs)

    actual_pred = obtain_actual_predictions(predicted, epsilon, n)
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
    if evaluation_verbosity == 1:
        print("tp: {}; fn: {}; tn: {}; fp: {}".format(tp, fn, tn, fp))
        print("test data size: {} positives; {} negatives; total: {}".format(positives, negatives, positives + negatives))
    if evaluation_verbosity >= 1:
        print("score: {}".format(9*tp + tn))

    return 9*tp + tn


def obtain_actual_predictions(predicted, epsilon, n):
    dfg = round_predictions_distance_from_greatest(epsilon, predicted)
    ofg = greatest_n(predicted, n)

    actual_predictions = list()
    for i in range(len(predicted)):
        if ofg[i] == dfg[i] == 1:
            actual_predictions.append(1)
        else:
            actual_predictions.append(0)
    # counter = 0
    # for p in actual_pred:
    #     counter += p
    # if counter < 10:
    #     return greatest_10(predicted)

    return actual_predictions


def greatest_n(predicted, n):
    largest = np.array(predicted)
    largest = sorted(list(largest))
    # print(largest)
    largest = largest[len(largest)-n:]
    # print(largest)

    actual_pred = list()
    for p in predicted:
        if p in largest:
            actual_pred.append(1)
        else:
            actual_pred.append(0)
    return actual_pred


def round_predictions_distance_from_greatest(epsilon, predicted):
    greatest = 0;
    for p in predicted:
        if p > greatest:
            greatest = p
    actual_pred = list()
    for p in predicted:
        if greatest - p < epsilon:
            actual_pred.append(1)
        else:
            actual_pred.append(0)
    return actual_pred


def main():
    print("initializing dc...")
    dc = DataCenter()
    dc.initialize(ngrams_size=ngram_size)

    all_users_ngrams = dc.all_data['all_users_session_ngrams_processed']

    training_set = list()
    for user in all_users_ngrams:
        training_set.append(user[:50])

    # num_of_anns = len(all_users_ngrams)
    anns = list()

    if _train_anns:
        print("generating anns... ({})".format(num_of_anns))
        generate_anns(anns, dc)
        print("done.")
        print("training anns... ({})".format(num_of_anns))
        i = 0;
        for user_ngrams, ann in zip(training_set, anns):
            train_ann(ann, i, training_set, user_ngrams, not _prediction_classifiers_late_training, dc)
            if _save_anns:
                print("saving ann {}...".format(i+1))
                save_model(ann, _model_save_path + str(i), overwrite=True)
            i += 1
    elif _load_anns:
        for i in range(num_of_anns):
            print("loading ann {}...".format(i + 1))
            anns.append(load_model(_model_save_path + str(i)))
    print("done.")

    if _evaluate:
        scores = list()
        print("for (i = {}; i < {}; i += {}) evaluate(i)".format(eval_epsilon, eval_epsilon + eval_step_size * num_of_eval_steps, eval_step_size))
        ep = eval_epsilon
        n = greatestn_n
        opt_ep, opt_n, max_score = ep, n, 0
        for i in range(num_of_eval_steps):
            current_score = evaluate_anns(anns, dc, ep, n)
            scores.append(current_score)
            if current_score > max_score:
                opt_ep = ep
                opt_n = n
                max_score = current_score
            ep += eval_step_size
            # n += 1
        print(",".join([str(s) for s in scores]))

    if _predict:
        print("predicting with ep={}, n={} (opt, score={}):".format(opt_ep, opt_n, max_score))
        predict_unknowns(anns, dc, training_set, opt_ep, opt_n)
        print("predicting with ep={}, n={} (score={}):".format(eval_epsilon, opt_n, scores[0]))
        predict_unknowns(anns, dc, training_set, opt_ep, opt_n)

    return


def predict_unknowns(anns, dc, training_set, epsilon, n):
    for i in range(num_of_tested_users, 40):
        if _prediction_classifiers_late_training:
            train_ann(anns[i], i, training_set, training_set[i], _prediction_classifiers_late_training, dc)
        labeled_sessions = dc.all_data['labeled'][i]
        xs = [ls['data'] for ls in labeled_sessions][50:]
        predictions = obtain_actual_predictions(anns[i].predict(np.array(xs)), epsilon=epsilon, n=n)
        print(",".join([str(p) for p in predictions]))


def evaluate_anns(anns, dc, epsilon, n):
    if evaluation_verbosity >= 1:
        print("evaluating... ({})".format(num_of_tested_users))
    average_score = 0;
    i = 0
    for i, ann in zip(range(num_of_tested_users), anns):
        labeled_sessions = dc.all_data['labeled'][i]
        xs = [ls['data'] for ls in labeled_sessions][50:]
        ys = [ls['label'] for ls in labeled_sessions][50:]
        average_score += evaluate_classifier(ann, np.array(xs), np.array(ys), epsilon, n)
        i += 1
    average_score = average_score / i
    # print("average score: {}".format(average_score))
    return average_score


def generate_anns(anns, dc):
    nok = len(dc.all_data['all_users_substitutions'].keys())
    prototype = generate_classifier(num_of_keys=nok, ratio=hidden_layer_to_input_size_ratio)
    for i in range(num_of_anns):
        # print("input_img size = {}".format(nok))
        # print("generating {} out of {}".format(i + 1, num_of_anns))
        ann = copy.deepcopy(prototype)
        anns.append(ann)


def train_ann(ann, i, training_set, user_ngrams, _train_prediction_classifier, dc):
    print("{} {}/{}".format(encoder_training_start_printout_prefix, i + 1, num_of_anns))
    if (i < num_of_tested_users and _evaluate) or (i >= num_of_tested_users and _train_prediction_classifier):
        x_train, y_train = list(), list()
        for user_sessions, j in zip(training_set, range(len(training_set))):
            if j != i:
                x_train.extend(user_sessions)
        # if i >= 10:
        #     labeled_sessions = dc.all_data['labeled'][i]
        #     x_train.extend([ls['data'] for ls in labeled_sessions][50:])
        random.shuffle(x_train)
        # x_train = x_train[:450]
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


if __name__ == "__main__":
    # for j in range(10):
    for i in range(8, 20, 1):
        negative_example_clones = i
        print("{} clones = {} {}".format(encoder_training_start_printout_prefix, i, encoder_training_start_printout_prefix))
        print("epochs = 3")
        epochs_number = 3
        main()
        print("epochs = 5")
        epochs_number = 5
        main()
        print("epochs = 4")
        epochs_number = 4
        main()
        print("epochs = 2")
        epochs_number = 2
        main()
        print("epochs = 1")
        epochs_number = 1
        main()
        # hidden_layer_to_input_size_ratio *= 2
