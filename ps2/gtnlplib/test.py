import numpy as np
from collections import defaultdict
import gtnlplib.preproc
import gtnlplib.preproc_metrics

import gtnlplib.clf_base
import gtnlplib.wordlist
import gtnlplib.naivebayes
import gtnlplib.perceptron
import gtnlplib.avg_perceptron
import gtnlplib.logreg

import gtnlplib.scorer
import gtnlplib.constants
import gtnlplib.analysis

poswords, negwords = gtnlplib.wordlist.loadSentimentWords(gtnlplib.constants.SENTIMENT_FILE)

weights_wlc = gtnlplib.wordlist.learnWLCWeights(poswords, negwords)
outfile = 'word_list.txt'
mat = gtnlplib.clf_base.evalClassifier(weights_wlc, outfile, gtnlplib.constants.DEVKEY)
print gtnlplib.scorer.printScoreMessage(mat)
