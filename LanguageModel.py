import re
import pickle
import numpy as np
import tqdm
from functools import lru_cache
from collections import Counter

class NGrammModel:
    def __init__(self, n):

        self.laplace_lambda = 0.98
        self.statistic_counter = Counter()
        self.count = 0 #всего n-gramm
        self.n = n

    def fit(self, queries):
        for tokens in queries:
            self.fit_tokens(tokens)

    def fit_tokens(self, tokens):
        self.statistic_counter.update(tokens)
        #self.count_ngramm += len(tokens)

    def token_prob(self, token):
        prob = self.laplace_lambda * (self.statistic_counter[token] / self.count)
        prob += (1 - self.laplace_lambda) / len(self.statistic_counter)

        return prob

    def tokens_prob(self, tokens):
        p_query = np.prod([self.token_prob(token) for token in tokens])

        return p_query

    def count_ngramm(self):
        self.count = 0
        del_list = []
        for ngram in self.statistic_counter:
            self.count += self.statistic_counter[ngram]
            if self.statistic_counter[ngram] == 1:
                del_list.append(ngram)

        for ngram in del_list:
            del self.statistic_counter[ngram]

    def tokens_log_likehood(self, tokens):
        query_log_likehood = np.sum(np.log([self.token_prob(token) for token in tokens]))

        return query_log_likehood

    def tokens_cross_entropy(self, tokens):
        query_cross_entropy = -self.tokens_log_likehood(tokens) / len(list(tokens))

        return query_cross_entropy

    def tokens_perplexity(self, tokens):
        return 2 ** self.tokens_cross_entropy(tokens)


class LanguageModel:
    def __init__(self):
        self.counter_frequency = Counter() #счетчик запросов
        self.unigram_model = NGrammModel(1)
        self.bigram_model = NGrammModel(2)
        self.bigram_char_model = NGrammModel(2)
        self.trigram_char_model = NGrammModel(3)

        self.query_count = 0

    def save(self):
        f = open('lm.pcl', 'wb')
        pickle.dump(self, f)
        f.close()

    def fit(self, queries, queries_tokens=None):
        #for i, (query, query_tokens) in tqdm.tqdm(enumerate(zip(queries, queries_tokens))):
        for i, query in enumerate(queries):
            if len(query) > 100:
                continue

            if queries_tokens is None:
                tokens = re.findall(r'\w+', query)
            else:
                tokens = queries_tokens[i]

            ## frequency
            self.counter_frequency.update([query])

            bigram_tokens = self.bigram_tokens(tokens)
            bichar_tokens, trichar_tokens = self.char_ngram_tokens(query)

            ## unigrams model statistic
            self.unigram_model.fit_tokens(tokens)

            ## bigrams model statistic
            self.bigram_model.fit_tokens(bigram_tokens)

            #bigrams char model statistic
            self.bigram_char_model.fit_tokens(bichar_tokens)

            #trigrams char model statistic
            self.trigram_char_model.fit_tokens(trichar_tokens)

            self.query_count += 1

        self.unigram_model.count_ngramm()
        self.bigram_model.count_ngramm()
        self.bigram_char_model.count_ngramm()
        self.trigram_char_model.count_ngramm()

        del_list = []
        for query in self.counter_frequency:
            if self.counter_frequency[query] == 1:
                del_list.append(query)

        for query in del_list:
            del self.counter_frequency[query]


    def bigram_tokens(self, query_tokens):
        bigram_tokens = ('<s>',) + tuple(query_tokens) + ('</s>',)
        bigram_tokens = zip(bigram_tokens, bigram_tokens[1:])
        return bigram_tokens

    def char_ngram_tokens(self, query):
        padded_query = "~~" + query + "~~" # ~~ is paddiing

        bichar_tokens = zip(padded_query[1:-1], padded_query[2:-1])
        trichar_tokens = zip(padded_query, padded_query[1:], padded_query[2:])

        return bichar_tokens, trichar_tokens

    def char_bigram_tokens(self, query):
        padded_query = "~" + query + "~" # ~~ is paddiing
        bichar_tokens = zip(padded_query[1:-1], padded_query[2:-1])

        return bichar_tokens

    def char_trigram_tokens(self, query):
        padded_query = "~~" + query + "~~" # ~~ is paddiing
        trichar_tokens = zip(padded_query, padded_query[1:], padded_query[2:])
        return trichar_tokens

    def query_frequences(self, query):
        return -np.log((self.counter_frequency[query] + 1)/ (self.query_count + 1 ))

    @lru_cache(maxsize=128)
    def P_query(self, query, ngramm_model='unigram', tokens_=None):
        if ngramm_model=='unigram':
            model = self.unigram_model
            tokens = tokens_ if tokens_ else re.findall(r'\w+', query)
        elif ngramm_model=='bigram':
            model = self.bigram_model
            if tokens_ is None:
                tokens = self.bigram_tokens(re.findall(r'\w+', query))
            else:
                tokens = self.bigram_tokens(tokens_)
        elif ngramm_model=='bigram_char':
            model = self.bigram_char_model
            tokens = self.char_bigram_tokens(query)
        elif ngramm_model=='trigram_char':
            model = self.trigram_char_model
            tokens = self.char_trigram_tokens(query)
        else:
            raise ValueError(f'ngramm_model argument: {ngramm_model} incorrect')

        return self.P_query_with_tokens(tokens, model)

    def P_query_with_tokens(self, tokens, model):
        tokens = list(tokens)
        log_likehood = model.tokens_log_likehood(tokens)
        cross_entropy = -log_likehood / max(len(tokens), 1)
        return log_likehood, cross_entropy

    @lru_cache(maxsize=128)
    def __call__(self, query):
        if self.query_count == 0:
            raise RuntimeError('Model not fitted')
        if isinstance(query, str):
            tokens = re.findall(r'\w+', query)
        elif isinstance(query, list):
            tokens = query
        bigrams = self.bigram_tokens(tokens)
        bichar_tokens, trichar_tokens = self.char_ngram_tokens(query)
        #bichar_tokens, trichar_tokens = list(bichar_tokens), list(trichar_tokens)

        query_frequency = self.query_frequences(query)

        P_query_unigrams = self.P_query_with_tokens(tokens, self.unigram_model)

        P_query_bigrams = self.P_query_with_tokens(bigrams, self.bigram_model)

        P_query_bigrams_char = self.P_query_with_tokens(bichar_tokens, self.bigram_char_model)

        P_query_trigrams_char = self.P_query_with_tokens(trichar_tokens, self.trigram_char_model)

        return [query_frequency, #частота запроса
                *P_query_unigrams,                                 #P(query) - unigram model
                *P_query_bigrams,                                  #P(query) - bigram model
                *P_query_bigrams_char,                             #P(query) - bigram char model
                *P_query_trigrams_char                             #P(query) - trigram char model
               ]

if __name__ == '__main__':
    lm = LanguageModel()
    lm.fit(queries, queries_token)
