import random

import math

from DataCenter import *
from ann import generate_classifier, ngram_simple_classifier

ngram_size = 2
batch_size = 1
# split_threshold = -1
rounding_threshold = 0.5
# feature_ratio_threshold = 0.2
epochs_number = 100
# num_of_extra_evaluations = 0
hidden_layer_to_input_size_ratio = 0.2

encoder_training_start_printout_prefix = "---------------------------------------------------"


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

    training_set = list()
    for user in all_users_ngrams:
        training_set.append(user[:50])

    # num_of_autoencoders = len(all_users_ngrams)
    num_of_autoencoders = 40
    print("generating auto encoders... ({})".format(num_of_autoencoders))
    autoencoders = list()
    for i in range(num_of_autoencoders):
        nok = len(dc.all_data['all_users_substitutions'][i].keys())
        print("input_img size = {}".format(nok))
        ae = generate_user_autoencoder_classifier(num_of_keys=nok, hidden_layer_size=int(hidden_layer_to_input_size_ratio * nok))
        autoencoders.append(ae)

    print("training auto encoders... ({})".format(num_of_autoencoders))
    i = 1;
    for user_ngrams, ae in zip(training_set, autoencoders):
        print("{} {}/{}".format(encoder_training_start_printout_prefix, i, num_of_autoencoders))
        i += 1
        user_ngrams = np.array(user_ngrams)
        ae.fit(user_ngrams, user_ngrams, epochs=epochs_number, batch_size=batch_size, verbose=2, shuffle=True)

    num_of_tested_users = 10
    print("evaluating... ({})".format(num_of_tested_users))
    ae_delta_prediction_threshold = 4.85
    scores = list()
    max_score = 0
    opt_threshold = 0
    print("evaluating")
    for k in range(50):
        average_score = 0;
        print("{}: threshold={}".format(k, ae_delta_prediction_threshold))
        for i in range(num_of_tested_users):
            labeled_sessions = dc.all_data['labeled'][i]
            xs = [ls['data'] for ls in labeled_sessions][50:]
            ys = [ls['label'] for ls in labeled_sessions][50:]
            average_score += evaluate_autoencoder(autoencoders[i], xs, ys, ae_delta_prediction_threshold)
        average_score = average_score/num_of_tested_users
        scores.append(average_score)
        if average_score >= max_score:
            max_score = average_score
            opt_threshold = ae_delta_prediction_threshold

        ae_delta_prediction_threshold += 0.001

    print("average scores: {}".format(scores))
    print("max score: {} ; opt threshold: {}".format(max_score, opt_threshold))

    for i in range(num_of_tested_users, 40):
        labeled_sessions = dc.all_data['labeled'][i]
        xs = [ls['data'] for ls in labeled_sessions][50:]
        predictions = ae_predictions(autoencoders[i], xs, opt_threshold)
        print(",".join([str(p) for p in predictions]))

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
