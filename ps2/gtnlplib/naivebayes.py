import numpy as np  # hint: np.log
from itertools import chain
from collections import defaultdict, Counter
from gtnlplib.preproc import dataIterator
from gtnlplib.constants import OFFSET, TRAINKEY, DEVKEY
from gtnlplib import scorer
from gtnlplib.clf_base import evalClassifier

''' keep the shell '''


def learnNBWeights(counts, class_counts, allkeys, alpha=0.1):
    weights = defaultdict(int)
    words_in_label = dict()
    total_instances = float(sum(class_counts.values()))

    for label, counter in counts.iteritems():
        words_in_label[label] = sum(counter.values())

    for word in allkeys:
        for label, counter in counts.iteritems():
            weights[label, word] = np.log((counter[word] + alpha) / (words_in_label[label] + alpha * len(allkeys)))

    for label, count in class_counts.iteritems():
        weights[label, OFFSET] = np.log(count / total_instances)

    return weights


def regularization_using_grid_search(alphas, counts, class_counts, allkeys, tr_outfile='nb.alpha.tr.txt',
                                     dv_outfile='nb.alpha.dv.txt'):
    tr_accs = []
    dv_accs = []
    # Choose your alphas here
    weights_nb_alphas = dict()
    for alpha in alphas:
        weights_nb_alphas[alpha] = learnNBWeights(counts, class_counts, allkeys, alpha)
        confusion = evalClassifier(weights_nb_alphas[alpha], tr_outfile, TRAINKEY)
        tr_accs.append(scorer.accuracy(confusion))
        confusion = evalClassifier(weights_nb_alphas[alpha], dv_outfile, DEVKEY)
        dv_accs.append(scorer.accuracy(confusion))
    return weights_nb_alphas, tr_accs, dv_accs
