from skanfis.fs import *
from skanfis import *
from sklearn.tree import export_text
from sklearn.tree import _tree
from sklearn import tree
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import KBinsDiscretizer
import itertools as it
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV
from collections import defaultdict, OrderedDict


def extract_crisp_rules(tree, feature_names, class_names=None, k=4):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    # print(feature_name)
    fns = []
    for item in feature_name:
        if item not in fns and item != 'undefined!':
            fns.append(item)

    crisp_rules = []
    path = []

    def recurse(node, path, paths):

        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            # print(node)
            # exit(0)
            threshold = tree_.threshold[node]
            tmp_threshold = np.round(threshold, 3)
            p1, p2 = list(path), list(path)
            if any(name in ti for ti in p1):
                for item in range(len(p1)):
                    if name in p1[item]:
                        # print(name)
                        # print(p1[item])
                        if '>' in p1[item]:
                            if '>=' not in p1[item]:
                                tmp_p1 = p1[item].split()
                                if float(tmp_threshold) > float(tmp_p1[2]):
                                    p1[item] = f"{tmp_threshold} >= " + p1[item]
                            else:
                                tmp_p1 = p1[item].split()
                                if float(tmp_p1[0]) < float(tmp_threshold):
                                    tmp_p1[0] = str(tmp_threshold)
                                    p1[item] = ' '.join(tmp_p1)
                        if '<=' in p1[item]:
                            p1[item] = f"{name} <= {tmp_threshold}"
                        # print(p1[item])
                        # exit(0)
                        break
            else:
                p1 += [f"{name} <= {tmp_threshold}"]
            recurse(tree_.children_left[node], p1, paths)

            if any(name in ti for ti in p2):
                for item in range(len(p2)):
                    if name in p2[item]:
                        # print(name)
                        if '<=' in p2[item]:
                            if '< ' not in p2[item]:
                                tmp_p2 = p2[item].split()
                                if float(tmp_threshold) < float(tmp_p2[2]):
                                    p2[item] = f"{tmp_threshold} < " + p2[item]
                            else:
                                tmp_p2 = p2[item].split()
                                # print(tmp_p2)
                                if float(tmp_p2[0]) > float(tmp_threshold):
                                    tmp_p2[0] = str(tmp_threshold)
                                    p2[item] = ' '.join(tmp_p2)
                        if '>' in p2[item]:
                            p2[item] = f"{name} > {tmp_threshold}"
                        # print(p1[item])
                        # exit(0)
                        break
            else:
                p2 += [f"{name} > {tmp_threshold}"]
            recurse(tree_.children_right[node], p2, paths)
        else:
            path += [(tree_.value[node], tree_.n_node_samples[node])]
            paths += [path]

    recurse(0, path, crisp_rules)
    # print(path)
    # print(paths)
    return crisp_rules, fns

def convert_into_fuzzy_rules_from_crisp(paths, class_names=None, k=4):
    # sort by samples count
    samples_count = [p[-1][1] for p in paths]
    ii = list(np.argsort(samples_count))
    paths = [paths[i] for i in reversed(ii)]
    # print(paths)

    def get_fuzzy_value_one(oper, strvalue):
        tmp_values = []
        for i in range(k):
            if eval(str(i) + oper + strvalue):
                tmp_values.append('mf'+str(i))
        return ','.join(tmp_values)
        # return tmp_values

    def get_fuzzy_value_two(strvalue1, oper1, oper2, strvalue2):
        tmp_values = []
        for i in range(k):
            if eval(strvalue1 + oper1 + str(i)) and eval(str(i) + oper2 + strvalue2):
                tmp_values.append('mf'+str(i))
        return ','.join(tmp_values)
        # return tmp_values

    fuzzy_rules = []
    for path in paths:
        # rule = "IF "
        mfs = []
        # print(len(path)-1)

        for p in range(len(path[:-1])):
            # if rule != "IF ":
            #     rule += " AND "
            ps = path[p].split()
            # print(ps)
            if len(ps) == 3:
                ps[2] = get_fuzzy_value_one(ps[1], ps[2])
                mfs.append(list(ps[2].split(',')))
                ps[1] = 'IS'
                path[p] = ' '.join(ps)
            # exit(0)
            if len(ps) == 5:
                ps[0] = get_fuzzy_value_two(ps[0], ps[1], ps[3], ps[4])
                mfs.append(ps[0].split(','))
                path[p] = ps[2] + ' IS ' + ps[0]
            # print(ps)
            # rule += '(' + path[p] + ')'
        # print(path)

        # print(mfs)
        for e in it.product(*mfs):
            # print(len(e))
            # print(e)
            rule = "IF "
            for p in range(len(path[:-1])):
                # print(p)
                # exit(0)
                if rule != "IF ":
                    rule += " AND "
                ps = path[p].split()
                # print(ps)
                ps[2] = e[p]
                ps = ' '.join(ps)
                # print(ps)
                # exit(0)

                rule += '(' + ps + ')'
            rule += " THEN "

            if class_names is None:
                    # print(path[-1])
                rule += "response: " + str(np.round(path[-1][0][0][0], 3))
                    # exit(0)
            else:
                classes = path[-1][0][0]
                # print(classes)
                l = np.argmax(classes)
                    # print(l)
                    # exit(0)
                rule += f"(y0 IS {class_names[l]})"
            fuzzy_rules += [rule]

    return fuzzy_rules




def convert_into_fs_by_scikit_anfis(initial_fuzzy_rules, X, inputs, _bin, _cn):
    # get the initial parameters (mu & sigma) of each gaussian membership function based on K
    # X = df[inputs]
    # print(inputs)
    # print(_fs._variables)
    X = X.values
    gX = X
    scaler = MinMaxScaler()
    gX = scaler.fit_transform(gX)
    est = KBinsDiscretizer(n_bins=_bin, encode='ordinal', strategy='uniform', subsample=None)
    gX = est.fit_transform(gX)

    vgs = {}
    for j in range(len(inputs)):
        tmp = []
        gms = X[:, j]
        for i in range(_bin):
            tgmss = gms[np.where(gX[:, j] == i)]
            mu = np.mean(tgmss)
            sigma = np.std(tgmss)
            tmp.append(tuple((mu, sigma)))
        vgs[inputs[j]] = tmp
    # print(vgs)
    # assert len(inputs) == len(vgs), 'input len is {}, vgs len is {}'.format(len(inputs), len(vgs))

    # create a fuzzy system
    _fs = FS()
    # print(_fs._variables)
    if len(_fs._variables) > 0:
        _fs._lvs = OrderedDict()
        _fs._variables = OrderedDict()
        # _lvs[self._variable]
    for name, value in vgs.items():
        S_tmp = []
        for j in range(len(value)):
            S_tmp.append(GaussianFuzzySet(mu=value[j][0], sigma=value[j][1], term="mf" + str(j)))
        _fs.add_linguistic_variable(name, LinguisticVariable(S_tmp))

    if len(_fs._variables) == 0:
        for k in _fs._lvs.keys():
            _fs._variables[k] = float(1)

    # print(_fs._lvs)
    # print(_fs._variables)

    for i in range(len(_cn)):
        _fs.set_crisp_output_value(_cn[i], i)
    _fs.add_rules(initial_fuzzy_rules)
    return _fs
