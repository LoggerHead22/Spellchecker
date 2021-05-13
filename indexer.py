import pickle
import time
import itertools
import re
import numpy as np
from LanguageModel import LanguageModel
from ErrorModel import ErrorModel
from Generators import JoinGenerator, SplitGenerator
from FuzzySearch import FuzzySearch
from ru_soundex.soundex import RussianSoundex
from collections import Counter, defaultdict


def process_queries(filename):
    with open(filename, encoding = 'utf-8') as f:
        queries = []
        queries_correction = []
        for line in f:
            line_s = line.strip().split('\t')
            if len(line_s[0]) < 100 and not line_s[0].isdigit():
                if len(line_s) == 1:
                    queries.append(line_s[0].lower())
                elif len(line_s) == 2:
                    queries.append(line_s[1].lower())
                    queries_correction.append([line_s[0].lower(),
                                               line_s[1].lower()])

    return queries, queries_correction


def fit_models(alpha_em=1.2, oper_weights=[0.7, 1.1, 0.9, 0.8]):
    queries, queries_correction = process_queries('queries_all.txt')

    print('Fitting language model...')
    t1 = time.process_time()
    lm = LanguageModel()
    lm.fit(queries)
    #lm.save()
    print(f'Language model fitted, time: {time.process_time() - t1}, query count: {lm.query_count}')

    print('Fitting Error Model...')
    t1 = time.process_time()
    em = ErrorModel(alpha_em, oper_weights)
    em.fit(queries_correction)
    #em.save()
    print('Error model fitted, time:', time.process_time() - t1)

    with open('lan_err_models.pcl', 'wb') as f:
        pickle.dump((lm, em), f)

    fs = FuzzySearch(0.02, lm, em)
    fs.fit(queries)

    print('Fitting Soundex collection')
    t1 = time.process_time()
    words = itertools.chain(*map(lambda x: re.findall(r'\w+', x),  queries))
    soundex = RussianSoundex(delete_first_letter=True)
    soundex_dict = dict()
    for word in words:
        if len(word) > 2 and len(word) < 20 and word.isalpha():
            try:
                code = soundex.transform(word)
                if code not in soundex_dict:
                    soundex_dict[code] = Counter([word])
                else:
                    soundex_dict[code].update([word])
            except:
                #print('Error word:', word)
                continue
    del_list = []
    for key in soundex_dict:
        if key.isupper():
            del_list.append(key)
        else:
            soundex_dict[key] = soundex_dict[key].most_common(5)

    for key in del_list:
        del soundex_dict[key]

    print('Soundex collection fitted, time:', time.process_time() - t1)

    with open('soundex_trie.pcl', 'wb') as f:
        pickle.dump((soundex_dict, fs.trie), f, pickle.HIGHEST_PROTOCOL)




def sample_random_words(queries):
    query_uno = []
    query_double = []

    np.random.seed(42)
    for query in queries:
        tokens = re.findall(r'\w+', query)
        if np.random.random() > 0.7:
            if len(tokens) > 2:
                i = np.random.randint(0, len(tokens) - 1)
                if not tokens[i].isdigit() and not tokens[i + 1].isdigit():
                    query_double.append(tokens[i: i+2])

        if np.random.random() > 0.01:
            if len(tokens) >=1:
                i = np.random.randint(0, len(tokens))
                if not tokens[i].isdigit():
                    query_uno.append(tokens[i])

    query_uno = np.unique(query_uno)
    query_double = np.unique(query_double, axis=0)

    return query_uno, query_double

if __name__ == '__main__':
    fit_models()


