from sklearn.preprocessing import MinMaxScaler
from Levenshtein import editops, distance
from collections import defaultdict, Counter
import numpy as np
import pickle

class ErrorModel:
    def __init__(self, alpha=1.1, oper_weights=[0.6, 0.8, 0.9, 0.8]):
        self.alpha = alpha

        self.oper_weights = {'replace': oper_weights[0],
                            'insert': oper_weights[1],
                            'delete': oper_weights[2],
                            'transpose': oper_weights[3]
                            }

        self.statistic = {'replace': Counter(),
                          'insert': Counter(),
                          'delete': Counter(),
                          'transpose': Counter()
                          }
        self.weights = {}

    def save(self):
        f = open('em.pcl', 'wb')
        pickle.dump(self, f)
        f.close()

    def def_lambda(self):
        return 1.

    def build_default_dict(self, key, def_value, values):

        ddict = defaultdict(self.oper_weight_lambda[key], values)
        return ddict

    def fit(self, queries):

        for query in queries:
            result = self.collect_stat_from_edit(query)
            for operation in result:
                self.statistic[operation].update(result[operation])

        for key in self.statistic:
            keys, values = list(self.statistic[key].keys()), list(self.statistic[key].values())
            scaler = MinMaxScaler((self.oper_weights[key] - 0.2, self.oper_weights[key] + 0.2))
            probs = scaler.fit_transform(-np.log(np.array(values).reshape(-1, 1) + 0.1))

            #self.weights[key] = self.build_default_dict(key,
            #                                            max(self.oper_weights[key] + 0.2, 1),
            #                                            zip(keys, list(probs.ravel())))

            self.weights[key] = defaultdict(self.def_lambda, zip(keys, list(probs.ravel())))


    def probability(self, orig, fix):
        return self.alpha ** (-float(self.dameraulevenshtein(orig, fix)))

    def collect_stat_from_edit(self, query):
        edit_operation = editops(query[0].lower(), query[1].lower())
        result = defaultdict(lambda: defaultdict(int))
        #print(edit_operation)
        s1 = '~' + query[0].lower()
        s2 = '~' + query[1].lower()

        oper_last, ind_1_last, ind_2_last = edit_operation[0]
        ind_1_last, ind_2_last = ind_1_last + 1, ind_2_last + 1

        if oper_last == 'delete':
            result[oper_last][(s1[ind_1_last - 1: ind_1_last + 1], s1[ind_2_last - 1: ind_2_last] + '_')] += 1
        elif oper_last == 'insert':
            result[oper_last][(s1[ind_1_last - 1: ind_1_last] + '_', s1[ind_1_last-1] + s2[ind_2_last])] += 1
        elif oper_last == 'replace':
            result[oper_last][(s1[ind_1_last - 1: ind_1_last + 1], s2[ind_2_last - 1: ind_2_last + 1])] += 1

        for i in range(1, len(edit_operation)):
            oper_cur, ind_1_cur, ind_2_cur = edit_operation[i]
            ind_1_cur, ind_2_cur = ind_1_cur + 1, ind_2_cur + 1

            if oper_cur == oper_last == 'replace' \
                and ind_1_last + 1 == ind_1_cur  \
                and ind_2_last + 1 == ind_2_cur \
                and s1[ind_1_last] == s2[ind_2_cur] \
                and s1[ind_1_cur] == s2[ind_2_last]:
                result['transpose'][(s1[ind_1_cur - 1: ind_1_cur + 1], s2[ind_2_cur - 1: ind_2_cur + 1])] += 1
            else:
                if oper_cur == 'delete':
                    result[oper_cur][(s1[ind_1_cur - 1: ind_1_cur + 1], s1[ind_1_cur - 1: ind_1_cur] + '_')] += 1
                elif oper_cur == 'insert':
                    result[oper_cur][(s1[ind_1_cur - 1: ind_1_cur] + '_', s1[ind_1_cur-1] + s2[ind_2_cur])] += 1
                elif oper_cur == 'replace':
                    result[oper_cur][(s1[ind_1_cur - 1: ind_1_cur + 1], s2[ind_2_cur - 1: ind_2_cur + 1])] += 1
                else:
                    assert True

            oper_last, ind_1_last, ind_2_last = oper_cur, ind_1_cur, ind_2_cur

        return result

    def dameraulevenshtein(self, seq1, seq2):
        weights = self.weights
        seq1 = '~' + seq1
        seq2 = '~' + seq2

        oneago = None
        thisrow = list(range(1, len(seq2) + 1)) + [0]
        for x in range(len(seq1)):
            twoago, oneago, thisrow = oneago, thisrow, [0] * (len(seq2)) + [x + 1]

            for y in range(len(seq2)):
                delcost = oneago[y] + weights['delete'][(seq1[x-1: x+1], seq1[x] + '_')]
                addcost = thisrow[y - 1] + weights['insert'][(seq1[x] + '_', seq1[x] + seq2[y])]
                subcost = oneago[y - 1] + weights['replace'][(seq1[x-1: x+1], seq2[y-1: y+1])] * (seq1[x] != seq2[y])

                thisrow[y] = min(delcost, addcost, subcost)

                # This block deals with transpositions
                if (x > 0 and y > 0 and seq1[x] == seq2[y - 1]
                    and seq1[x-1] == seq2[y] and seq1[x] != seq2[y]):
                    thisrow[y] = min(thisrow[y], twoago[y - 2] + weights['transpose'][(seq1[x-1: x+1], seq2[y-1: y+1])])

        return float(thisrow[len(seq2) - 1])
