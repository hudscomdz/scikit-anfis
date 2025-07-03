# from skanfis.fs import *
from .fs import *

def rule_antec_parsing(func, varnames, varlist, mf_indices, i, operators):
    if type(func) == Functional:
        # print(x[0]._A._variable, x[0]._A._term, x[0]._fun, x[0]._B._variable, x[0]._B._term)
        # print(x[0]) #f.(c.(x0 IS mf0) AND_p f.(c.(x1 IS mf0) AND_p f.(c.(x2 IS mf0) AND_p c.(x3 IS mf0))))

        col1 = varnames.index(func._A._variable)
        value1 = list(varlist[col1].names()).index(func._A._term)
        mf_indices[i][col1] = value1
        operators.append(func._fun)

        if type(func._B) == Functional:  # todo
            rule_antec_parsing(func._B, varnames, varlist, mf_indices, i, operators)

        else:
            # print(func)
            # print(varnames)
            # print(varlist)
            col2 = varnames.index(func._B._variable)
            value2 = list(varlist[col2].names()).index(func._B._term)
            mf_indices[i][col2] = value2

    else:
        # print(x[0]._variable, x[0]._term)
        col = varnames.index(func._variable)
        value = list(varlist[col].names()).index(func._term)
        mf_indices[i][col] = value
        operators.append('NONE')


def rule_recursive_parse(rules, varnames, varlist):
    # results = [float(antecedent[0].evaluate(self)) for antecedent in self._rules]
    mf_indices = [[-1 for col in range(len(varlist))] for row in range(len(rules))]
    operators = []
    # print(varnames)
    # print(mf_indices)

    # a = ['{} is {} {} {} is {}'.format(x[0]._A._variable, x[0]._A._term, x[0]._fun, x[0]._B._variable, x[0]._B._term) if type(
    #     x[0]) == Functional else '{} is {}'.format(x[0]._variable, x[0]._term) for x in rules]
    i = -1
    for x in rules:
        i = i + 1
        # print(type(x[0]))
        rule_antec_parsing(x[0], varnames, varlist, mf_indices, i, operators)
        # if type(x[0]) == Functional:
        #     # print(x[0]._A._variable, x[0]._A._term, x[0]._fun, x[0]._B._variable, x[0]._B._term)
        #     # print(x[0]) #f.(c.(x0 IS mf0) AND_p f.(c.(x1 IS mf0) AND_p f.(c.(x2 IS mf0) AND_p c.(x3 IS mf0))))
        #
        #     col1 = varnames.index(x[0]._A._variable)
        #     value1 = list(varlist[col1].names()) .index(x[0]._A._term)
        #
        #     if type(x[0]._B) == Functional: #todo
        #         rule_recursive_parse(x[0]._B, varnames, varlist)
        #
        #     col2 = varnames.index(x[0]._B._variable)
        #     value2 = list(varlist[col2].names()).index(x[0]._B._term)
        #     mf_indices[i][col1] = value1
        #     mf_indices[i][col2] = value2
        #     # print(type(x[0]._fun))
        #     operators.append(x[0]._fun)
        # else:
        #     # print(x[0]._variable, x[0]._term)
        #     col = varnames.index(x[0]._variable)
        #     value = list(varlist[col].names()).index(x[0]._term)
        #     mf_indices[i][col] = value
        #     operators.append('NONE')

    # print(mf_indices)
    # print(operators)
    # exit(0)

    return mf_indices, operators
