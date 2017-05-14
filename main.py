import random

import math

from DataCenter import *
from ann import generate_classifier, ngram_simple_classifier

ngram_size = 2
batch_size = 1
# split_threshold = -1
# rounding_threshold = 0.5
# feature_ratio_threshold = 0.2
epochs_number = 3
# num_of_extra_evaluations = 0
hidden_layer_to_input_size_ratio = 0.2

encoder_training_start_printout_prefix = "---------------------------------------------------"


def show_ratios(fraud, normal, keys):
    mal_counter = dict()
    ben_counter = dict()
    ratios_dict = dict()

    already_counted = dict()
    informative_keys = list()

    for cmd in keys:
        mal_counter[cmd] = ben_counter[cmd] = 0;
        already_counted[cmd] = -1

    index = 0;

    for section in normal:
        for cmd in section:
            if already_counted[cmd] != index:
                ben_counter[cmd] += 1;
                index = index + 1
                already_counted[cmd] = index

    for section in fraud:
        for cmd in section:
            if already_counted[cmd] != index:
                mal_counter[cmd] += 1;
                index = index + 1
                already_counted[cmd] = index

    for cmd in keys:
        mal_counter[cmd] = mal_counter[cmd] / len(fraud)
        ben_counter[cmd] = ben_counter[cmd] / len(normal)
        ratios_dict[cmd] = (mal_counter[cmd] + 1) / (ben_counter[cmd] + 1)

    for cmd in keys:
        if abs(ratios_dict[cmd] - 1) > feature_ratio_threshold:
            print(cmd + " <--  " + str(ratios_dict[cmd]))
            informative_keys.append(cmd)

    return informative_keys


def vectorize_list(data, key_to_index):
    new_data = list()
    n = len(key_to_index.keys())
    for section in data:
        vec = np.zeros(n)
        for word in section:
            if word in key_to_index.keys():
                vec[key_to_index[word]] += 1.0
        new_data.append(vec)
    return new_data


def make_x_y_lists(fraud, benign):
    x = []
    y = []
    x.extend(benign)
    y.extend(np.zeros(len(benign)))
    x.extend(fraud)
    y.extend(np.ones(len(fraud)))

    x = np.array(x)
    y = np.array(y)
    return x, y


def ae_predictions(ae, xs, threshold):
    final_predictions = list();

    deltas = list()
    reconstructions = ae.predict(np.array(xs))
    for x, r in zip(xs, reconstructions):
        delta_vec = [(vx - vr)**2 for vx,vr in zip(x, r)]
        delta = 0
        for d in delta_vec:
            delta += d
        delta = math.sqrt(delta)
        deltas.append(delta)

    for d in deltas:
        if d > threshold:
            final_predictions.append(1)
        else:
            final_predictions.append(0)
        # print("delta: {}".format(d))

    return final_predictions


def evaluate_autoencoder(ae, xs, ys, threshold):

    tp = tn = fp = fn = 0
    predicted = ae_predictions(ae, xs, threshold)
    # print(predicted)
    for i in range(len(xs)):
        if ys[i] == 0 and predicted[i] == 0: tn += 1
        elif ys[i] == 0 and predicted[i] == 1: fp += 1
        elif ys[i] == 1 and predicted[i] == 0: fn += 1
        elif ys[i] == 1 and predicted[i] == 1: tp += 1

    print("tp: {}; fn: {}; tn: {}; fp: {}".format(tp, fn, tn, fp))
    print("score: {}".format(9*tp + fp))
    print("test data size: {} positives; {} negatives; total: {}".format(tp+fn, tn+fp, tp+fn+tn+fp))

    return 9*tp + tn


def main():
    print("initializing dc...")
    dc = DataCenter()
    dc.initialize(ngrams_size=ngram_size)

    normal_sessions = dc.all_data['raw'][:dc.num_of_benign_sessions_per_user]
    all_users_ngrams = dc.all_data['all_users_session_ngrams_processed']

    training_set = [user[:50] for user in all_users_ngrams]
    testing_set = [user[50:] for user in all_users_ngrams[:10]]
    unknown_set = [user[50:] for user in all_users_ngrams[10:]]

    testing_labels = [[ls['label'] for ls in user] for user in dc.all_data['labeled'][:10]]
    testing_set = dc.flatten_list(testing_set)
    testing_labels = dc.flatten_list(testing_labels)

    print("generating auto encoders...")
    nok = len(dc.all_data['substitution'].keys())
    ae = generate_user_autoencoder_classifier(num_of_keys=nok, hidden_layer_size=int(hidden_layer_to_input_size_ratio * nok))

    print("training auto encoder...")
    training_set = np.array(dc.flatten_list(training_set))
    ae.fit(training_set, training_set, epochs=epochs_number, batch_size=batch_size, verbose=1, shuffle=True)

    num_of_tested_users = 10
    print("evaluating over {} users)".format(num_of_tested_users))
    ae_delta_prediction_threshold = 0
    scores = list()
    max_score = 0
    opt_threshold = 0
    for k in range(50):
        print("{}: threshold={}".format(k, ae_delta_prediction_threshold))
        count = 0
        for l in testing_labels:
            count += l
        print("counter = {}".format(count))
        score = evaluate_autoencoder(ae=ae, xs=testing_set, ys=testing_labels, threshold=ae_delta_prediction_threshold)
        scores.append(score)
        if score >= max_score:
            max_score = score
            opt_threshold = ae_delta_prediction_threshold

        ae_delta_prediction_threshold += 0.2

    print("average scores: {}".format(scores))
    print("max score: {} ; opt threshold: {}".format(max_score, opt_threshold))

    for user in unknown_set:
        xs = [ls['data'] for ls in user]
        predictions = ae_predictions(ae, xs, opt_threshold)
        print(",".join([str(p) for p in predictions]))

    return


if __name__ == "__main__":
    main()
