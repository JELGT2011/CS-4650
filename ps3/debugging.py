import numpy as np
from collections import defaultdict
from collections import defaultdict, Counter

import gtnlplib.preproc
import gtnlplib.viterbi
import gtnlplib.most_common
import gtnlplib.naivebayes
import gtnlplib.clf_base
import gtnlplib.scorer
import gtnlplib.constants
import gtnlplib.tagger_base
import matplotlib.pyplot as plt


# Define the file names
training_file = gtnlplib.constants.TRAIN_FILE
development_file = gtnlplib.constants.DEV_FILE
test_file = gtnlplib.constants.TEST_FILE  # You do not have this for now
offset = gtnlplib.constants.OFFSET

# Demo
all_tags = set()
for i, (words, tags) in enumerate(gtnlplib.preproc.conllSeqGenerator(training_file)):
    for tag in tags:
        all_tags.add(tag)
print "all_tags = " + str(all_tags)

counters = gtnlplib.most_common.get_tags(training_file)
for tag, tag_ctr in counters.iteritems():
    print tag, tag_ctr.most_common(3)
