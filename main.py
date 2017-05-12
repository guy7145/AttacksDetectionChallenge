import random

import math

from DataCenter import *
from ann import generate_classifier, ngram_simple_classifier

ngram_size = 2
batch_size = 1
# split_threshold = -1
rounding_threshold = 0.5
# feature_ratio_threshold = 0.2
epochs_number = 60
# num_of_extra_evaluations = 0
hidden_layer_to_input_size_ratio = 0.3
ae_delta_prediction_threshold = 1

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


def split_data(fraud, benign, train_to_total_ratio):
    fraud = list(fraud)
    benign = list(benign)
    random.shuffle(fraud)
    random.shuffle(benign)

    nf = len(fraud)
    nb = len(benign)

    train_f = fraud[:int(nf * train_to_total_ratio) - 1]
    test_f = fraud[int(nf * train_to_total_ratio):]
    train_b = benign[:int(nb * train_to_total_ratio) - 1]
    test_b = benign[int(nb * train_to_total_ratio):]

    return train_f, test_f, train_b, test_b


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


def round_predictions(predicted_rough):
    predicted = list()
    for p in predicted_rough:
        if p > rounding_threshold:
            predicted.append(1)
        else:
            predicted.append(0)
    return predicted


def evaluate(fraud, benign, key_to_index, unknown):
    fraud = vectorize_list(fraud, key_to_index)
    benign = vectorize_list(benign, key_to_index)
    unknown = vectorize_list(unknown, key_to_index)

    if split_threshold != -1:
        (train_f, test_f, train_b, test_b) = split_data(fraud, benign, split_threshold)
        x_train, y_train = make_x_y_lists(train_f, train_b)
        x_test, y_test = make_x_y_lists(test_f, test_b)
    else:
        x_train, y_train = make_x_y_lists(fraud, benign)
        x_test, y_test = make_x_y_lists(fraud, benign)

    classifier = ngram_simple_classifier(len(key_to_index.keys()))
    # print("(training...)")
    classifier.fit(x_train, y_train, epochs=epochs_number, batch_size=batch_size, verbose=0, shuffle=True)
    # print("(evaluating...)")
    total_loss = classifier.evaluate(x_test, y_test, batch_size=3000, verbose=0)

    print("evaluation: " + str(total_loss) + "")

    tp = tn = fp = fn = 0
    predicted = round_predictions(classifier.predict(x_test))

    # print(predicted)
    for i in range(len(x_test)):
        if y_test[i] == 0 and predicted[i] == 0:
            tn += 1
        elif y_test[i] == 0 and predicted[i] == 1:
            fn += 1
        elif y_test[i] == 1 and predicted[i] == 0:
            fp += 1
        elif y_test[i] == 1 and predicted[i] == 1:
            tp += 1

    print("tp: {}; fn: {}; tn: {}; fp: {}".format(tp, fn, tn, fp))
    print("test data size: {} positives; {} negatives; total: {}".format(tp+fn, tn+fp, tp+tn+fp+fn))

    return classifier, unknown


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
    negatives = positives = 0
    # print(predicted)
    for i in range(len(xs)):
        if ys[i] == 0 and predicted[i] == 0:
            tn += 1
            negatives += 1
        elif ys[i] == 0 and predicted[i] == 1:
            fp += 1
            negatives += 1
        elif ys[i] == 1 and predicted[i] == 0:
            fn += 1
            positives += 1
        elif ys[i] == 1 and predicted[i] == 1:
            tp += 1
            positives += 1

    print("tp: {}; fn: {}; tn: {}; fp: {}".format(tp, fn, tn, fp))
    print("score: {}".format(9*tp + fp))
    print("test data size: {} positives; {} negatives; total: {}".format(positives, negatives, positives + negatives))

    return 9*tp + tn


def main():
    print("initializing dc...")
    dc = DataCenter()
    dc.initialize(ngrams_size=ngram_size)

    normal_sessions = dc.all_data['raw'][:dc.num_of_benign_sessions_per_user]
    all_users_ngrams = dc.all_data['all_users_session_ngrams_processed']

    # num_of_autoencoders = len(all_users_ngrams)
    num_of_autoencoders = 10
    print("generating auto encoders... ({})".format(num_of_autoencoders))
    autoencoders = list()
    for user_ngrams, i in zip(all_users_ngrams, range(num_of_autoencoders)):
        nok = len(dc.all_data['all_users_substitutions'][i].keys())
        print("input_img size = {}".format(nok))
        ae = generate_user_autoencoder_classifier(num_of_keys=nok, hidden_layer_size=int(hidden_layer_to_input_size_ratio * nok))
        autoencoders.append(ae)

    print("training auto encoders... ({})".format(num_of_autoencoders))
    i = 1;
    for user_ngrams, ae in zip(all_users_ngrams, autoencoders):
        print("{} {}/{}".format(encoder_training_start_printout_prefix, i, num_of_autoencoders))
        i += 1
        user_ngrams = np.array(user_ngrams)
        ae.fit(user_ngrams, user_ngrams, epochs=epochs_number, batch_size=batch_size, verbose=2, shuffle=True)

    num_of_tested_users = 10
    print("evaluating... ({})".format(num_of_tested_users))
    ae_delta_prediction_threshold = 0.75
    scores = list()
    max_score = 0
    opt_threshold = 0
    print("evaluating")
    for k in range(20):
        average_score = 0;
        print("{}: threshold={}".format(k, ae_delta_prediction_threshold))
        for i in range(num_of_tested_users):
            labeled_sessions = dc.all_data['labeled'][i]
            xs = [ls['data'] for ls in labeled_sessions]
            ys = [ls['label'] for ls in labeled_sessions]
            average_score += evaluate_autoencoder(autoencoders[i], xs, ys, ae_delta_prediction_threshold)
        average_score = average_score/num_of_tested_users
        scores.append(average_score)
        if average_score >= max_score:
            max_score = average_score
            opt_threshold = ae_delta_prediction_threshold

        ae_delta_prediction_threshold += 0.0025

    print("average scores: {}".format(scores))
    print("max score: {} ; opt threshold: {}".format(max_score, opt_threshold))

    return

    # for user_sessions in dc.all_data['all_users_session_ngrams_processed']:
    #     print(user_sessions)


    # print("loading data...")
    # dc.load_all_raw_data()
    # print("generating substitution...")
    # dc.generate_substitution()
    # print("loading labels...")
    # dc.load_labels(dc.raw_data)
    #
    # # feature extraction
    # print("generating ngrams (" + str(ngram_size) + ")")
    # dc.generate_self_ngrams(ngram_size)
    # informative_keys = show_ratios(dc.labeled_ngrams_fraud, dc.labeled_ngrams_normal, dc.all_ngrams)
    # key_to_index = dict()
    # for i in range(len(informative_keys)):
    #     key_to_index[informative_keys[i]] = i
    #
    # # print(key_to_index)
    # for i in range(num_of_extra_evaluations):
    #     evaluate(dc.labeled_ngrams_fraud, dc.labeled_ngrams_normal, key_to_index, dc.labeled_ngrams_unknown)
    #
    # (classifier, unknown) = evaluate(dc.labeled_ngrams_fraud, dc.labeled_ngrams_normal, key_to_index, dc.labeled_ngrams_unknown)
    #
    # predictions = round_predictions(classifier.predict(np.array(unknown)))
    #
    # csv = [','.join(np.array(predictions[i:i+100]).astype(str)) for i in range(30)]
    # csv = '\n'.join(csv)
    # print(csv)

if __name__ == "__main__":
    main()