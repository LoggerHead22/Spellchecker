from sklearn.ensemble import RandomForestClassifier
from pyxdameraulevenshtein import damerau_levenshtein_distance
import pickle
from LanguageModel import LanguageModel
from ErrorModel import ErrorModel

class Classifier():
    def __init__(self, lm, em, process_func=None):
        self.model = RandomForestClassifier()
        self.lan_model = lm
        self.err_model = em

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, query):
        features = self.process_func(query)
        prediction = self.model.predict_proba(features)
        return prediction



    def features_to_split(self, query):
        features = np.array(self.lan_model(token_1) + [len(query)]).reshape((1, -1))
        return features


class FixNoneClassifier:
    def __init__(self, lm, em):
        self.model = RandomForestClassifier()
        self.lan_model = lm
        self.err_model = em

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, orig, fix):
        features =  self.features_to_fix_none(query_orig, query_fix)
        prediction = self.model.predict_proba(features)
        return prediction

    def features_to_fix_none(self, query_orig, query_fix):
        lm_features_orig = np.array(self.lan_model(query_orig))
        lm_features_fix = np.array(self.lan_model(query_fix))
        lm_features_diff = lm_features_orig - lm_features_fix
        damerau_dist_weighted = self.err_model.probability(query_orig, query_fix)
        damerau_dist = damerau_levenshtein_distance(query_orig, query_fix)
        prob = em.alpha ** (-damerau_dist_weighted)
        len_diff = len(query_orig) - len(query_fix)

        em_features = np.array([damerau_dist_weighted, damerau_dist,
                                prob, len_diff])

        result = np.concatenate((lm_features_orig, lm_features_fix,
                            lm_features_diff, em_features)).reshape((1, -1))


