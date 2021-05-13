import itertools
import heapq
from Trie import Trie
import re
import numpy as np

class FuzzySearch:
    def __init__(self, alpha, lm, em):
        self.alpha = alpha
        self.trie = Trie()
        self.err_model = em
        self.lan_model = lm
        self.queue = []

    def fit(self, queries):
        print('Building trie...')
        for query in queries:
            words = re.findall(r'\w+', query)
            for token in words:
                if token.isdigit() == False and len(token) < 25:
                    self.trie.add(str(token))
        print('Cleaning trie...')
        self.del_garbage(self.trie.root_node)
        print('Computing trie freq...')
        self.compute_prefix_freq(self.trie.root_node, prefix='')

    def generate_candidates(self, orig):
        q_last = [(-self.weight(orig, '', self.lan_model.P_query('', 'trigram_char')[1] * -1),
                   ("", "", self.trie.root_node))]

        orig_prefix = ""
        flag = True
        result = []

        k = max(min(2, len(orig) - 1), 1)
        #print('K', k)

        iteration=0
        while flag:
            iteration += 1
            q_cur = []
            flag = False

            for w, (orig_prefix, fix_prefix, node) in q_last:
                if len(orig_prefix) == 2 and  len(orig) == 1:
                    print(orig_prefix, fix_prefix)

                if len(orig_prefix) < len(orig):
                    flag = True
                    self.gen_on_node(q_cur, w, orig_prefix + orig[len(orig_prefix)], fix_prefix, node) #замена

                    if len(orig_prefix) + k >= len(fix_prefix):
                        self.gen_on_node(q_cur, w, orig_prefix, fix_prefix, node) #вставка

                    if len(orig_prefix) + 1 < len(orig) and len(orig_prefix) - k <= len(fix_prefix):
                        self.gen_on_node(q_cur, w, orig_prefix + orig[len(orig_prefix): len(orig_prefix) + 2],
                                      fix_prefix, node) #удаление

                else:
                    if len(orig_prefix) + k >= len(fix_prefix):
                        self.gen_on_node(q_cur, w, orig_prefix, fix_prefix, node) #вставка в конец

                    if node.p_unigram < 16 and np.abs(len(orig) -  len(fix_prefix)) <= k:
                        heapq.heappush(result, (float(w), node.p_unigram, fix_prefix))
                    #result[fix_prefix] = w

            #print(len(q_cur))
            #if iteration <= 2:
            q_last = heapq.nsmallest(min(iteration * 10, 40), q_cur)
            #else:
            #q_last = heapq.nsmallest(50, q_cur)

        return heapq.nsmallest(15, result)

    def weight(self, pref_orig, pref_fix, freq):
        #w = self.alpha * self.lan_model(pref_fix)[-3] * -1
        w = self.alpha * freq
        #print(pref_orig, pref_fix, self.alpha * freq, np.log(self.err_model.probability(pref_orig, pref_fix)))
        #      self.alpha * freq +  np.log(em.probability(pref_orig, pref_fix)))
        w += np.log(self.err_model.probability(pref_orig, pref_fix))
        return w

    def gen_on_node(self, q, weight_old, orig_prefix, fix_prefix, node):
        for prefix in node.child_nodes:
            cur_node = node.child_nodes[prefix]
            weight = self.weight(orig_prefix, fix_prefix + prefix, cur_node.freq)
            #print(orig_prefix, fix_prefix + prefix, weight_old, -weight)
            #if weight_old >= -weight - 0.2:
            heapq.heappush(q, (-weight, (orig_prefix, fix_prefix + prefix, cur_node)))

    def compute_prefix_freq(self, node, prefix=''):
        total_count = 0
        for key in node.child_nodes:
            total_count += node.child_nodes[key].flag

        for key in node.child_nodes:
            node.child_nodes[key].freq = self.lan_model.P_query(prefix + key, 'trigram_char')[1] * -1
            node.child_nodes[key].p_unigram = self.lan_model.P_query(prefix + key, 'unigram')[1]
            #node.child_nodes[key].freq = node.child_nodes[key].flag / total_count
            self.compute_prefix_freq(node.child_nodes[key], prefix + key)

    def del_garbage(self, node):
        del_list = []

        for key in node.child_nodes:
            if not (ord(key) >= 65 and ord(key) <= 122 or re.findall('[а-яА-ЯёЁ]+', key)):
            #if not (re.findall('[а-яА-ЯёЁ]+', key)):
                del_list.append(key)

        for key in del_list:
            del node.child_nodes[key]

        if node.child_nodes:
            for key in node.child_nodes:
                self.del_garbage(node.child_nodes[key])

if __name__ == '__main__':
    pass
