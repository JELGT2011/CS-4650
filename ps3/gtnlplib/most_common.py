
import operator
from  constants import *
from collections import defaultdict, Counter
import preproc
import scorer
import constants
import clf_base
import csv
from nltk.tokenize import word_tokenize, sent_tokenize

argmax = lambda x: max(x.iteritems(), key=operator.itemgetter(1))[0]


def get_tags(trainfile):
    """Produce a Counter of occurences of word in each tag"""
    counts = dict()

    for i, (words, tags) in enumerate(preproc.conllSeqGenerator(trainfile)):
        for tag in tags:
            counts[tag] = Counter()
            for word in words:
                counts[tag][word] = 0

    with open(trainfile) as tsv:
        for line in csv.reader(tsv, delimiter="\t", quoting=csv.QUOTE_NONE):
            if len(line) == 2:
                word = line[0]
                tag = line[1]
                counts[tag][word] += 1

    tsv.close()

    return counts


def get_noun_weights():
    """Produce weights dict mapping all words as noun"""
    return your_weights


def get_most_common_weights(trainfile):
    return weights


def get_class_counts(counters):
    return counts
