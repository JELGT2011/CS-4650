
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

import tests.testpreproc as test

test.setup_module()
test.test_number_of_tokens_in_training()
test.test_token_type_ratio_for_dev()
test.test_token_type_ratio_for_train()
test.test_type_frequency_for_dev()
test.test_type_frequency_for_train()
test.test_unseen_types()

# gtnlplib.preproc.docsToBOWs(gtnlplib.constants.TRAINKEY)
# gtnlplib.preproc.docsToBOWs(gtnlplib.constants.DEVKEY)

# ac_train = gtnlplib.preproc.getAllCounts(gtnlplib.preproc.dataIterator(gtnlplib.constants.TRAINKEY))
# ac_dev = gtnlplib.preproc.getAllCounts(gtnlplib.preproc.dataIterator(gtnlplib.constants.DEVKEY))


