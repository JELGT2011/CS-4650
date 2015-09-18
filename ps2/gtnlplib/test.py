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

counts, class_counts, allkeys = gtnlplib.preproc.getCountsAndKeys(gtnlplib.constants.TRAINKEY)
weights_nb = gtnlplib.naivebayes.learnNBWeights(counts, class_counts, allkeys, alpha=0.1)

outfile = 'nb.txt'
mat = gtnlplib.clf_base.evalClassifier(weights_nb, outfile, gtnlplib.constants.DEVKEY)
print gtnlplib.scorer.printScoreMessage(mat)
