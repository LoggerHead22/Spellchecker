import re
import numpy as np
import heapq
from itertools import chain
from collections import defaultdict
from LanguageModel import LanguageModel
from ErrorModel import ErrorModel
from Trie import Trie
from FuzzySearch import FuzzySearch
from functools import lru_cache
from sklearn.ensemble import RandomForestClassifier
from ru_soundex.soundex import RussianSoundex
from loguru import logger
import snoop
import heartrate
#heartrate.trace(browser=True)

class KeyboardSwapper:
    def __init__(self, lm):
        keyboard_ru = "йцукенгшщзхъфывапролджэячсмитьбю."
        keyboard_en = "qwertyuiop[]asdfghjkl;'zxcvbnm,./"
        keyboard_dig = r'1234567890-+!@#$%^&*()_=`~ё'

        zip_ = chain(zip(keyboard_en, keyboard_ru), zip(keyboard_ru, keyboard_en),
                     zip(keyboard_dig, keyboard_dig))

        self.en_ru = defaultdict(lambda: '', {pair[1] : pair[0] for pair in zip_})
        self.lan_model = lm


    def token_swap(self, token):
        #if len(token) <=1:
        #    return token

        result = ""
        token = token.strip()
        for char in token:
            if char.isdigit():
                return token
            result += self.en_ru[char]

        if len(result) != len(token):
            return token

        P_result_unigram = self.lan_model.P_query(result, 'unigram')[0]
        P_token_unigram = self.lan_model.P_query(token, 'unigram')[0]

        score_1 = 1 / 3 * P_result_unigram
        score_2 = 1 / 3 * P_token_unigram

        score_1 += 2 / 3 *  self.lan_model.P_query(result, 'trigram_char')[0]
        score_2 += 2 / 3 * self.lan_model.P_query(token, 'trigram_char')[0]

        #print(token,'-->', result, round(score_1, 3), round(score_2, 3), round(score_1 / score_2, 3),
        #          round(self.lan_model(result)[1]  / self.lan_model(token)[1], 3))

        case_1 = score_1 > 0.75 * score_2 and \
            P_result_unigram > (0.8 - 0.2 / len(token) ** 2) * P_token_unigram

        case_2 = score_1 > 0.6 * score_2

        case_3 = P_result_unigram > 0.6 * P_token_unigram and P_token_unigram > 0

        case_4 = self.lan_model.query_frequences(result) < self.lan_model.query_frequences(token) and self.lan_model.query_frequences(token) < 10

        condition = int(case_1) + int(case_2) + int(case_3) + int(case_4)

        #print(token, int(case_1) + int(case_2) + int(case_3) + int(case_4))

        border = 0 if len(token) > 1 else 1

        if condition > border:
        #if case_1 or case_2 or case_3 or case_4:
            print(token,'-->', result, round(score_1, 3), round(score_2, 3), round(score_1 / score_2, 3),
                  round(self.lan_model(result)[1]  / self.lan_model(token)[1], 3))

            return result
        else:
            return token

    #@snoop
    def generate(self, query):
        result = []
        tokens = query.strip().split(' ')

        #if len(tokens) == 1 and len(tokens[0]) > 20:
        #    return query

        for token in tokens:
            result.append(self.token_swap(token))

        return result

class CorrectionGenerator:
    def __init__(self, lm, em, fs, soundex):
        self.lan_model = lm
        self.fuzzy_search = fs #FuzzySearch(0.02, 10, em, lm)
        self.soundex_dict = soundex
        self.soundex_encoder = RussianSoundex(delete_first_letter=True)

    @lru_cache(maxsize=128)
    def candidates(self, word, fs):
        res = fs.generate_candidates(word)
        soundex_code = self.soundex_encoder.transform(word)
        if soundex_code in self.soundex_dict:
            for candidate, count in self.soundex_dict[soundex_code]:
                candidate_weight = float(-fs.weight(word, candidate,
                                                    self.lan_model.P_query(candidate, 'trigram_char')[1] * -1))

                heapq.heappush(res, (candidate_weight,
                                     self.lan_model.P_query(candidate, 'unigram')[1],
                                     candidate))

        res = np.unique(res, axis=0)
        #print(word, '-->', res[:10])
        scores = res[:,0].astype('float') * res[:,1].astype('float')
        arg_scores = np.argsort(scores)

        return zip(scores[arg_scores], res[arg_scores, 2])

    #@logger.catch
    def generate(self, query, correction_mask): #query is tokens
        corrections = {}
        for index in np.where(correction_mask == 0)[0]:
            corrections[index] = self.candidates(query[index], self.fuzzy_search)

        fixed_query = self.find_best_pairs(corrections, query)

        return [fixed_query]


    def find_best_pairs(self, corrections, queries):
        fixed_query = [token for token in queries]

        correction_found = False

        for index in corrections:
            if index == 0:
                fixed_query[index] = next(corrections[index])[1] #если первое слово, то выбираем лучшее по весу
                correction_found = True
                #print('first word: ', fixed_query[index])
            else:
                score = self.lan_model.P_query(None, 'bigram', tuple(fixed_query[:index + 1]))[0] #без исправления такой вес
                best_word = None
                for weight, fixed_word in corrections[index]:

                    current_score = self.lan_model.P_query(None, 'bigram', tuple(fixed_query[:index] + [fixed_word]))[0]
                    #print(fixed_word, current_score)
                    if current_score > score:
                        score = current_score
                        best_word = fixed_word

                if best_word:
                    fixed_query[index] = best_word
                    correction_found = True
        #print('After finding pairs: ', fixed_query)

        if correction_found:
            return fixed_query
        else:
            return None


class JoinGenerator:
    def __init__(self, lm):
        self.lan_model = lm
        self.model = RandomForestClassifier(min_samples_split=10,
                                            min_samples_leaf=5,
                                            max_features='sqrt',
                                            max_samples=0.7)

    def fit(self, X, y):
        self.model.fit(X, y)


    def generate(self, tokens, correction_mask):
        new_tokens = [tokens[0]]
        for i, (token_1, token_2) in enumerate(zip(tokens, tokens[1:])):
            prob = self.predict(new_tokens[-1], token_2)

            if prob[0][1] > 0.9:
                new_tokens[-1] += token_2
            else:
                new_tokens.append(token_2)
        return new_tokens


    @lru_cache(maxsize=128)
    def predict(self, token_1, token_2):
        return self.model.predict_proba(self.features_to_join(token_1,
                                                              token_2))

    def build_split(self, query_uno):
        query_splitted = []
        for query in query_uno:
            if len(query) > 1 and np.random.random() > 0.9:
                i = np.random.randint(1, len(query))
                query_splitted.append([query[:i], query[i:]])
        return query_splitted

    def build_join_X(self, queries, splitted_queries):
        X = []
        y = np.array([0] * len(queries) + [1] * len(splitted_queries))
        for query in chain(queries, splitted_queries):
            X.append(self.features_to_join(query[0], query[1]))
        X = np.array(X)
        indx = np.random.permutation(np.arange(len(y)))

        return X[indx].reshape((X.shape[0], -1)), y[indx]


    @lru_cache(maxsize=128)
    def features_to_join(self, token_1, token_2):
        lm_features_1 = np.array(self.lan_model(token_1))
        lm_features_2 = np.array(self.lan_model(token_2))
        lm_features_join = np.array(self.lan_model(token_1 + token_2))

        lm_features_diff_1 = lm_features_join - lm_features_1
        lm_features_diff_2 = lm_features_join - lm_features_2

        lens_features = np.array([len(token_1), len(token_2),
                                  len(token_1) + len(token_2)])

        result = np.concatenate((lm_features_1, lm_features_2, lm_features_join,
                                 lm_features_diff_1, lm_features_diff_2,
                                 lens_features)).reshape((1, -1))
        return result


class SplitGenerator:
    def __init__(self, lm):
        self.lan_model = lm
        self.model = RandomForestClassifier(min_samples_leaf=5,
                                            max_features='sqrt',
                                            max_samples=0.7)
    def fit(self, X, y):
        self.model.fit(X, y)

    def generate(self, tokens, joiner):
        new_tokens = []
        for token in tokens:
            if len(token) > 2:
                prob = self.predict(token)
                print(prob)
                if prob[0][1] > 0.5: #возможно надо делить
                    new_tokens += self.split_by_joiner(tokens, joiner)
                else:
                    new_tokens.append(token)
            else:
                new_tokens.append(token)

        return new_tokens


    def split_by_joiner(self, token, joiner):
        probs = []
        for i in range(2):
            probs.append(joiner.predict(token[:i + 1], token[i + 1:])[1])
            probs.append(joiner.predict(token[:-(i + 1)], token[-(i + 1):])[1])

        argmin = np.argmin(probs)
        print(prob)

        if probs[argmin] < 0.2: #мы почти уверены что соединять не надо, значит можно делить
            i = 1 + argmin >=2
            i *= -1 ** (argmin % 2)
            result = [token[:i], token[i:]]
        else:
            result = [token]

        return result


    @lru_cache(maxsize=128)
    def predict(self, token):
        return self.model.predict_proba(self.features_to_split(token))


    def build_join(self, query_double):
        query_joined = []
        for query in query_double:
            if np.random.random() > 0.9:
                query_joined.append("".join(query))
        return query_joined


    def build_split_X(self, queries, joined_queries):
        X = []
        y = np.array([0] * len(queries) + [1] * len(joined_queries))
        for query in chain(queries, joined_queries):
            X.append(self.features_to_split(query))
        X = np.array(X)
        indx = np.random.permutation(np.arange(len(y)))

        return X[indx].reshape((X.shape[0], X.shape[-1])), y[indx]


    @lru_cache(maxsize=128)
    def features_to_split(self, query):
        features = np.array(self.lan_model(query) + [len(query)]).reshape((1, -1))
        return features


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

#%%
if __name__ == '__main__':
    queries, queries_correction = process_queries('queries_all.txt')

    lm = LanguageModel()
    lm.fit(queries)

    em = ErrorModel()
    em.fit(queries_correction)


#%%
    with open('join_split_generators.pcl', 'rb') as f:
        join_gen, split_gen = pickle.load(f)
#%%
    print(split_gen.predict('вовсех'))
    print(split_gen.predict('сккачатьигру'))
    print(split_gen.predict('признание'))
    print(split_gen.predict('моямама'))
    print(split_gen.predict('невыдуманная'))
    print(split_gen.predict('сказкаи'))

#%%

    print(lm.P_query(None, 'bigram', tuple(['когда', 'рас', 'пускаются', 'почки']))[1])

    #split_gen_2 = SplitGenerator(lm)
    #split_gen_2.model = split_gen.model

    #print(split_gen_2.generate(['всех'], join_gen))
#%%
#    for query in queries:
        #new_query = join_gen.generate(query.split(' '))
#        new_query = split_gen_2.generate(query.split(' '), join_gen)
#        if new_query != query.split(' '):
#            print(query,'-->', new_query)

#%%
    key_swapper = KeyboardSwapper(lm)
    input_ = 'купить ноутбук asus'
    result_1 = key_swapper.generate(input_)
    result_2 = key_swapper.generate(" ".join(result_1))

    #assert(" ".join(result_2) == input_)
#%%
    print(key_swapper.en_ru['?'])


#%%
    print(key_swapper.generate('relf c[jlbnm gjuekznm'))
    print(key_swapper.generate('cjkjvf ytljhjuj'))
    print(key_swapper.generate('clfnm ht,tyrf d ghb.n'))
    print(key_swapper.generate('аднштп щт еру ьщщт'))
    print(key_swapper.generate('.n.,'))
    print(key_swapper.generate('мл'))

    print(result_1, result_2)
    print(lm('vk'))
    print(lm('мл'))
    print(lm('d'))
    print(lm('в'))

    print(key_swapper.generate('!(молодцы)'))

#%%
    for i in range(10000,30000):
        #if 'dj' in queries_tokens[i]:
        #    print(queries_tokens[i])
        res = key_swapper.generate(queries[i].lower())
        #print(queries[i], '--->',' '.join(res))


#%%

    frequency = []
    words = []
    for query in queries:
        if len(query) > 0:
            frequency.append(lm.query_frequences(query))
            if frequency[-1] < 10:
                words.append(query)

    for word in np.unique(words):
        #print(word, '--->', key_swapper.generate(word))
        key_swapper.generate(word)

