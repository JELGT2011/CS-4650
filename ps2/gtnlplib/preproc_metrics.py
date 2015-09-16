def get_token_type_ratio(vocabulary):
    return float(sum(vocabulary.values())) / len(vocabulary)


def type_frequency(vocabulary, k):
    # token[1] gets value after transformation of vocabulary into dictionary
    return len(filter(lambda token: token[1] == k, vocabulary.items()))


def unseen_types(first_vocab, second_vocab):
    # finds number of tokens in the first_vocab that are not in the second, AKA
    # first_vocab - second_vocab
    return len(set(second_vocab.keys()).difference(set(first_vocab.keys())))
