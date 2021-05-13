import pickle
import numpy as np
import heapq
import sys
from LanguageModel import LanguageModel
from ErrorModel import ErrorModel
from Generators import KeyboardSwapper, SplitGenerator, JoinGenerator
from Generators import CorrectionGenerator
from FuzzySearch import FuzzySearch

class Spellchecker:
    def __init__(self):

        self.generators = {}

        with open('lan_err_models.pcl', 'rb') as f:
            self.lan_model, self.err_model = pickle.load(f)

        #with open('join_split_generators.pcl', 'rb') as f:
        #     self.generators['join'], self.generators['split'] = pickle.load(f)


        self.keyboard_swapper = KeyboardSwapper(self.lan_model)
        self.fix_none_class = None

        self.fuzzy_search = FuzzySearch(0.02, self.lan_model, self.err_model)

        with open('soundex_trie.pcl', 'rb') as f:
            self.soundex, self.fuzzy_search.trie = pickle.load(f)

        self.generators['correction'] = CorrectionGenerator(self.lan_model,
                                                           self.err_model,
                                                           self.fuzzy_search,
                                                           self.soundex)
        self.generators['join'] = JoinGenerator(self.lan_model, self.soundex)
        self.generators['split'] = SplitGenerator(self.lan_model)

        self.iter_count = 3


    def process_query(self, query):
        query = query.strip()
        if len(query.strip()) <= 1 or len(query) > 100 or query.isdigit():     #если слишком длинный или которткий запрос то ничего не делаем
            return query

        query_swapped = self.keyboard_swapper.generate(query)  #меняем раскладку клавиатуры если нужно
        old_query = query_swapped
        #print('After swapper: ', old_query)

        if self.lan_model.query_frequences(" ".join(old_query)) < 10: #если запрос очень частотный скорее всего ошибки нет
            return " ".join(old_query)

        for i in range(self.iter_count): #
            new_query = self.iteration(old_query)
            if new_query is None:
                break
            else:
                #print(f'After iter {i}: ', new_query)
                old_query = new_query

        return " ".join(old_query)


    def iteration(self, query):
        correction_mask = np.array([self.lan_model.P_query(token,'unigram')[1] < 9 or token.isdigit() for token in query])
        if np.sum(correction_mask) == len(query):
            return None  #коррекция не нужна

        #print(correction_mask)
        fixed_queries = []
        p_bigrams = self.lan_model.P_query(None, 'bigram', tuple(query))[1]

        for generator in self.generators:

            fixed_query = self.generators[generator].generate(query, correction_mask)
            #print(generator,'--->',fixed_query)
            if fixed_query:
                fixed_queries += fixed_query

        if len(fixed_queries) == 0:
            return None

        #print(fixed_queries)
        final_fixes = []
        bit_mask = 0
        bin_array = np.zeros((len(fixed_queries)), dtype='int')
        candidates = np.where(np.bitwise_and(bin_array , bit_mask) == 0)[0] #исправления которые не пересекаются

        while len(candidates) > 0:
            scores = []
            for i  in candidates:

                i_1, i_2, fixed_word, bit = fixed_queries[i]
                bin_array[i] = bit #запоминаем позицию каждого исправления
                fixed_query = query[:i_1] + fixed_word + query[i_2:] #исправленный запрос
                scores.append(p_bigrams - self.lan_model.P_query(None, #тупой классификатор
                                                                 'bigram',
                                                                 tuple(fixed_query)
                                                                 )[1])
            #print(scores)
            argmax = np.argmax(scores)
            if scores[argmax] <= 0:
                break
            else:
                bit_mask |= bin_array[candidates[argmax]]
                heapq.heappush(final_fixes, fixed_queries[candidates[argmax]])
                candidates = np.where(np.bitwise_and(bin_array , bit_mask) == 0)[0]
                #print(bin(bit_mask), candidates)

        #print('Final fixes', final_fixes)

        if len(final_fixes) > 0:
            final_fixes.sort()
            new_query = []
            begin = 0
            for i_1, i_2, fixes, _ in final_fixes:
                new_query += query[begin:i_1]
                new_query += fixes
                begin = i_2

            new_query += query[i_2:]

            return new_query
        else:
            return None


    def start(self):
        for line in sys.stdin:
            print(self.process_query(line.strip()))

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
    spellchecker = Spellchecker()
    spellchecker.start()
    #queries, queries_correction = process_queries('queries_all.txt')

    #count = 0
    #for query in queries_correction[:300]:
    #    fixed_query = spellchecker.process_query(query[0])
    #    if query[1].strip() == fixed_query:
    #        count += 1
        #print(query[1])
    #print(f'Result: {count / 300}')