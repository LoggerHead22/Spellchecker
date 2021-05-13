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
        if len(res) == 0:
            return []

        res = np.unique(res, axis=0)
        #print(word, '-->', res[:10])
        scores = res[:,0].astype('float') * res[:,1].astype('float')
        arg_scores = np.argsort(scores)

        return list(zip(scores[arg_scores], res[arg_scores, 2]))

    #@logger.catch
    def generate(self, query, correction_mask): #query is tokens
        corrections = {}
        #print(query, np.where(correction_mask == 0)[0])
        for index in np.where(correction_mask == 0)[0]:
            #print(index)
            candidate = self.candidates(query[index], self.fuzzy_search)
            if candidate:
                corrections[index] = candidate

        #print(corrections)
        fixed_query = self.find_best_pairs(corrections, query)
        #print(fixed_query)
        return fixed_query


    def find_best_pairs(self, corrections, queries):

        fixed_query = [token for token in queries]
        fixed_candidates = []

        for index in corrections:
            if index == 0:
                fixed_candidates.append((index, index + 1, [corrections[index][0][1]],
                                         1 << (len(queries) - index - 1))) #если первое слово, то выбираем лучшее по весу
                #bit_mask |= 1 << (len(queries) - index - 1)
            else:
                score = self.lan_model.P_query(None, 'bigram', tuple(fixed_query[:index + 1]))[0] #без исправления такой вес
                best_word = None
                for weight, fixed_word in corrections[index]:

                    current_score = self.lan_model.P_query(None, 'bigram', tuple(fixed_query[:index] + [fixed_word]))[0]
                    #print(fixed_word, current_score, score)
                    if current_score > score:
                        score = current_score
                        best_word = fixed_word

                if best_word:
                    fixed_candidates.append((index, index + 1, [best_word], 1 << (len(queries) - index - 1))) #
                    #bit_mask |= 1 << (len(queries) - index - 1) # ставим 1 на месте исправленного слова


        return fixed_candidates


class JoinGenerator:
    def __init__(self, lm, soundex):
        self.lan_model = lm
        self.model = RandomForestClassifier(min_samples_split=10,
                                            min_samples_leaf=5,
                                            max_features='sqrt',
                                            max_samples=0.7)

        self.soundex_dict = soundex
        self.soundex_encoder = RussianSoundex(delete_first_letter=True)

    def fit(self, X, y):
        self.model.fit(X, y)


    def generate(self, tokens, correction_mask):
        fixed_candidates = []
        for i, (mask_1, mask_2) in enumerate(zip(correction_mask, correction_mask[1:])):
            if mask_1 and mask_2:
                continue

            soundex_code = self.soundex_encoder.transform(tokens[i] + tokens[i + 1])
            if soundex_code in self.soundex_dict:
                for candidate, count in self.soundex_dict[soundex_code][:2]:
                    fixed_candidates.append((i, i + 2, [candidate], 3 << (len(tokens) - i - 2)))

            condition = self.condition(tokens[i], tokens[i+1])

            if condition: #если вероятность джойна больше чем худшего из токенов
                fixed_candidates.append((i, i + 2,  [tokens[i] + tokens[i + 1]], 3 << (len(tokens) - i - 2)))

        return fixed_candidates

    @lru_cache(maxsize=128)
    def condition(self, token_1, token_2):
        p_1 = self.lan_model.P_query(token_1, 'unigram')[1]
        p_2 = self.lan_model.P_query(token_2, 'unigram')[1]
        p_3 = self.lan_model.P_query(token_1 + token_2, 'unigram')[1]

        return p_3 < max(p_1, p_2)


class SplitGenerator:
    def __init__(self, lm):
        self.lan_model = lm
        self.model = RandomForestClassifier(min_samples_leaf=5,
                                            max_features='sqrt',
                                            max_samples=0.7)
    def fit(self, X, y):
        self.model.fit(X, y)

    def generate(self, query, correction_mask): #query is tokens
        corrections = []
        for index in np.where(correction_mask == 0)[0]:
            corrections += self.candidates(query[index], index, len(query))

        return corrections

    def candidates(self, word, index, query_len):
        if len(word) == 1:
            return []

        fixed_candidates = []
        p_3 = self.lan_model.P_query(word, 'unigram')[1]

        for i in range(1, len(word) - 1):
            token_1 = word[:i]
            token_2 = word[i:]

            condition = self.condition(token_1, token_2, p_3)

            if condition: #если вероятность хб одной части больше чем всего токена
                fixed_candidates.append((index, index + 1,  [token_1, token_2], 1 << (query_len - index - 1)))

        return fixed_candidates

    @lru_cache(maxsize=128)
    def condition(self, token_1, token_2, p_3):
        p_1 = self.lan_model.P_query(token_1, 'unigram')[1]
        p_2 = self.lan_model.P_query(token_2, 'unigram')[1]

        return p_3 > min(p_1, p_2)



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


