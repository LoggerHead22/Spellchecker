import sys


def test():
    for line in sys.stdin:
        print(line.strip())
    print('By By')
    
if __name__ == '__main__':
    test()
    
    
    
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

    print(spellchecker.process_query('вайт и вв личний кбинет'))

#%%
    print(spellchecker.fuzzy_search.generate_candidates('смотреь'))
    print(spellchecker.lan_model('дорогие'))

    print(spellchecker.err_model.probability('вв','в'))
    print(spellchecker.err_model.probability('восьма','возьми'))
    print(spellchecker.err_model.dameraulevenshtein('смотреь','смотреть'))

#%%

    join = JoinGenerator(spellchecker.lan_model)

    print(join.generate(['пойд', 'м', 'в'], [0, 0, 1]))
    
    
    
   
   
   
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
