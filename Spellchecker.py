import pickle
import numpy as np
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
                print(f'After iter {i}: ', new_query)
                old_query = new_query

        return " ".join(old_query)


    def iteration(self, query):
        correction_mask = np.array([self.lan_model.P_query(token,'unigram')[1] < 9 or token.isdigit() for token in query])
        if np.sum(correction_mask) == len(query):
            return None  #коррекция не нужна

        fixed_queries = []
        p_bigrams = self.lan_model.P_query(None, 'bigram', tuple(query))[1]

        for generator in self.generators:
            fixed_query = self.generators[generator].generate(query, correction_mask)
            if fixed_query[0]:
                fixed_queries += fixed_query

        if len(fixed_queries) == 0:
            return None

        scores = []
        for fixed_query in fixed_queries:
            scores.append(p_bigrams - self.lan_model.P_query(None,
                                                             'bigram',
                                                             tuple(fixed_query)
                                                             )[1])
        #print(scores, fixed_queries)
        if np.max(scores) > 0:
            return fixed_queries[np.argmax(scores)]
        else:
            return None


    def start(self):
        pass
        #while query = input().strip():

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


if __name__ == '__main__':
    spellchecker = Spellchecker()

#%%
    #print(spellchecker.generators.keys())
    #print(spellchecker.generators['correction'].generate(['когда', 'рас', 'пускаются', 'почки']))

#%%
    #print(spellchecker.generators['correction'].generate(['когда', 'распускаются', 'почки']))
    #print(spellchecker.generators['correction'].generate(['когда', 'распускаются', 'почки']))
#%%
    #print(spellchecker.lan_model.P_query(None, 'bigram', ('когда', 'нас', 'пускаются', 'точки')))
    #print(spellchecker.lan_model.P_query(None, 'bigram', ('когда', 'распускаются', 'почки')))

#%%
    queries, queries_correction = process_queries('queries_all.txt')

    print(spellchecker.process_query(queries[0]))
#%%
    for query in queries[:10]:
        print('RESULT: ', query,'--->', spellchecker.process_query(query))


#%%
    print(spellchecker.process_query('мебель для ванной de aqua'))

#%%
    for query in queries_correction[:50]:
        print('RESULT: ', query[0],'--->', spellchecker.process_query(query[0]))
        print('Answer: ', query[1])

#%%

    print(spellchecker.process_query('тескт песни дениса клявера королева'))

#%%
    print(spellchecker.fuzzy_search.generate_candidates('дорогие'))
    print(spellchecker.lan_model('дорогие'))

    print(spellchecker.err_model.probability('восьма','восьми'))
    print(spellchecker.err_model.probability('восьма','возьми'))
    print(spellchecker.err_model.probability('восьма','весна'))

#%%
