#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import itertools
import math
import pandas
import re
from collections import OrderedDict
from .antecedent_parsing import rule_recursive_parse
from .experimental import *
from .membership import *
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.special import softmax

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


dtype = torch.float


class FuzzifyVariable(torch.nn.Module):
    '''
        Represents a single fuzzy variable, holds a list of its MFs.
        Forward pass will then fuzzify the input (value for each MF).
    '''
    def __init__(self, mfdefs):
        super(FuzzifyVariable, self).__init__()
        # print(type(mfdefs))

        if isinstance(mfdefs, dict):
            mfnames = [i for i in mfdefs.keys()]
            mfdefs = [eval(i[0])(*i[1]) for i in mfdefs.values()]
            mfdefs = OrderedDict(zip(mfnames, mfdefs))
            # print(mfdefs)
            # exit(0)

        if isinstance(mfdefs, list):  # No MF names supplied
            mfnames = ['mf{}'.format(i) for i in range(len(mfdefs))]
            mfdefs = OrderedDict(zip(mfnames, mfdefs))

        self.mfdefs = torch.nn.ModuleDict(mfdefs)
        self.padding = 0

    @property
    def num_mfs(self):
        '''Return the actual number of MFs (ignoring any padding)'''
        return len(self.mfdefs)

    def members(self):
        '''
            Return an iterator over this variables's membership functions.
            Yields tuples of the form (mf-name, MembFunc-object)
        '''
        return self.mfdefs.items()

    def names(self):
        return self.mfdefs.keys()

    def pad_to(self, new_size):
        '''
            Will pad result of forward-pass (with zeros) so it has new_size,
            i.e. as if it had new_size MFs.
        '''
        self.padding = new_size - len(self.mfdefs)

    def fuzzify(self, x):
        '''
            Yield a list of (mf-name, fuzzy values) for these input values.
        '''
        for mfname, mfdef in self.mfdefs.items():
            yvals = mfdef(x)
            yield(mfname, yvals)

    def forward(self, x):
        '''
            Return a tensor giving the membership value for each MF.
            x.shape: n_cases
            y.shape: n_cases * n_mfs
        '''
        y_pred = torch.cat([mf(x) for mf in self.mfdefs.values()], dim=1)
        if self.padding > 0:
            y_pred = torch.cat([y_pred,
                                torch.zeros(x.shape[0], self.padding)], dim=1)
        return y_pred


class FuzzifyLayer(torch.nn.Module):
    '''
        A list of fuzzy variables, representing the inputs to the FIS.
        Forward pass will fuzzify each variable individually.
        We pad the variables so they all seem to have the same number of MFs,
        as this allows us to put all results in the same tensor.
    '''
    def __init__(self, varmfs, varnames=None):
        super(FuzzifyLayer, self).__init__()
        # print(varnames)
        # exit(0)
        if not varnames:
            self.varnames = ['x{}'.format(i) for i in range(len(varmfs))]
        else:
            self.varnames = list(varnames)
        maxmfs = max([var.num_mfs for var in varmfs])
        for var in varmfs:
            var.pad_to(maxmfs)
        self.varmfs = torch.nn.ModuleDict(zip(self.varnames, varmfs))
        # print(self.varmfs) #print(self.varmfs)
        self._antec = {}


    @property
    def num_in(self):
        '''Return the number of input variables'''
        return len(self.varmfs)

    @property
    def max_mfs(self):
        ''' Return the max number of MFs in any variable'''
        return max([var.num_mfs for var in self.varmfs.values()])

    @property
    def antec(self):
        '''
            Output MFS parameters of the Input variables, i.e. antecedent
        '''
        # print('haha2')
        # ant = {}
        # # print(ant)
        # # print()
        for varname, members in self.varmfs.items():
            self._antec[varname] = {}
            for mfname, mfdef in members.mfdefs.items():
                # print(mfdef.pretty())
                self._antec[varname][mfname] = [p.item() for n, p in mfdef.named_parameters()]
                # r.append('- {}: {}({})'.format(mfname,
                #          mfdef.__class__.__name__,
                #          ', '.join(['{}={}'.format(n, p.item())
                #                    for n, p in mfdef.named_parameters()])))
        # print(self._antec)
        # exit(0)
        # return ant
        return self._antec


    def __repr__(self):
        '''
            Print the variables, MFS and their parameters (for info only)
        '''
        # print('haha1')
        r = ['Input variables']
        for varname, members in self.varmfs.items():
            r.append('Variable {}'.format(varname))
            for mfname, mfdef in members.mfdefs.items():
                r.append('- {}: {}({})'.format(mfname,
                         mfdef.__class__.__name__,
                         ', '.join(['{}={}'.format(n, p.item())
                                   for n, p in mfdef.named_parameters()])))
        return '\n'.join(r)


    def forward(self, x):
        ''' Fuzzyify each variable's value using each of its corresponding mfs.
            x.shape = n_cases * n_in
            y.shape = n_cases * n_in * n_mfs
        '''
        assert x.shape[1] == self.num_in,\
            '{} is wrong no. of input values {}'.format(self.num_in, x.shape[1])
        y_pred = torch.stack([var(x[:, i:i+1])
                              for i, var in enumerate(self.varmfs.values())],
                             dim=1)
        return y_pred


class AntecedentLayer(torch.nn.Module):
    '''
        Form the 'rules' by taking all possible combinations of the MFs
        for each variable. Forward pass then calculates the fire-strengths.
    '''
    def __init__(self, varlist, rules=None, varnames=None):
        super(AntecedentLayer, self).__init__()
        # Count the (actual) mfs for each variable:
        mf_count = [var.num_mfs for var in varlist]
        # print(varlist)
        # print(mf_count)
        # print(varnames)

        if not varnames:
            self.varnames = ['x{}'.format(i) for i in range(len(varlist))]
        else:
            self.varnames = varnames

        # Now make the MF indices for each rule:
        self.rules = rules
        if not rules:
            mf_indices = itertools.product(*[range(n) for n in mf_count])
            # print(list(mf_indices))
            operators = ['AND_p'] * math.prod(mf_count)
        else:
            # num_rule = len(rules)
            # mf_indices = [[-1 for col in range(len(varlist))] for row in range(len(rules))]
            # print(mf_indices) # n_rules * n_in (which mf index in the input)
            mf_indices, operators = rule_recursive_parse(self.rules, self.varnames, varlist)

        # print(list(mf_indices))

        self.mf_indices = torch.tensor(list(mf_indices))
        # print(self.mf_indices.shape)
        # mf_indices.shape is n_rules * n_in
        self.operators = list(operators)
        # print(self.operators)

        # print(torch.eq(self.mf_indices, -1))

    def num_rules(self):
        return len(self.mf_indices)

    def extra_repr(self, varlist=None):
        if not varlist:
            return None
        row_ants = []
        mf_count = [len(fv.mfdefs) for fv in varlist.values()]
        for rule_idx in itertools.product(*[range(n) for n in mf_count]):
            thisrule = []
            for (varname, fv), i in zip(varlist.items(), rule_idx):
                thisrule.append('{} is {}'
                                .format(varname, list(fv.mfdefs.keys())[i]))
            row_ants.append(' and '.join(thisrule))
        return '\n'.join(row_ants)

    def forward(self, x):
        ''' Calculate the fire-strength for (the antecedent of) each rule
            x.shape = n_cases * n_in * n_mfs
            y.shape = n_cases * n_rules
        '''
        # Expand (repeat) the rule indices to equal the batch size:
        # print(self.mf_indices.shape)
        batch_indices = self.mf_indices.expand((x.shape[0], -1, -1))
        # global tmprules
        if not self.rules:
            # Then use these indices to populate the rule-antecedents
            ants = torch.gather(x.transpose(1, 2), 1, batch_indices)
            # ants.shape is n_cases * n_rules * n_in
            # Last, take the AND (= product) for each rule-antecedent
            tmprules = torch.prod(ants, dim=2)
        else:
            # print('heihei')
            # print(self.operators)
            condition = torch.eq(batch_indices, -1)
            batch_indices = torch.where(condition, torch.zeros_like(batch_indices), batch_indices)
            # batch_indices.shape is n_cases * n_rules * n_in
            # print(batch_indices)
            ants = torch.gather(x.transpose(1, 2), 1, batch_indices)
            # print(ants)
            # ants.shape is n_cases * n_rules * n_in
            if 'OR' in self.operators:
                ants = torch.where(condition, torch.zeros_like(ants), ants)
                # print(ants)
                tmprules = torch.max(ants, dim=2)[0]
            elif 'AND_p' in self.operators:
                ants = torch.where(condition, torch.ones_like(ants), ants)
                tmprules = torch.prod(ants, dim=2)
            elif 'AND' in self.operators:
                ants = torch.where(condition, torch.ones_like(ants), ants)
                tmprules = torch.prod(ants, dim=2)
            else:
                ants = torch.where(condition, torch.ones_like(ants), ants)
                tmprules = torch.prod(ants, dim=2)
        return tmprules


class ConsequentLayer(torch.nn.Module):
    '''
        A simple linear layer to represent the TSK consequents.
        Hybrid learning, so use MSE (not BP) to adjust coefficients.
        Hence, coeffs are no longer parameters for backprop.
    '''
    def __init__(self, d_in, d_rule, d_out, zerotype=False):
        super(ConsequentLayer, self).__init__()
        self.zerotype = zerotype
        if self.zerotype:
            c_shape = torch.Size([d_rule, d_out, 1])
        else:
            c_shape = torch.Size([d_rule, d_out, d_in+1])
        self._coeff = torch.zeros(c_shape, dtype=dtype, requires_grad=True)
        # print(self._coeff.shape)
        # exit(0)

    @property
    def coeff(self):
        '''
            Record the (current) coefficients for all the rules
            coeff.shape: n_rules * n_out * (n_in+1)
        '''
        return self._coeff

    @coeff.setter
    def coeff(self, new_coeff):
        '''
            Record new coefficients for all the rules
            coeff: for each rule, for each output variable:
                   a coefficient for each input variable, plus a constant
        '''
        if not self.zerotype:
            assert new_coeff.shape == self.coeff.shape, \
                'Coeff shape should be {}, but is actually {}'\
                .format(self.coeff.shape, new_coeff.shape) #todo
        self._coeff = new_coeff

    def fit_coeff(self, x, weights, y_actual):
        '''
            Use LSE to solve for coeff: y_actual = coeff * (weighted)x
                  x.shape: n_cases * n_in
            weights.shape: n_cases * n_rules
            [ coeff.shape: n_rules * n_out * (n_in+1) ]
                  y.shape: n_cases * n_out
        '''
        # print('scheduling the fit_coeff')
        # print(self._coeff)
        # print(weights.shape)
        # exit(0)
        # Append 1 to each list of input vals, for the constant term:
        # x = torch.zeros(x.shape)
        if self.zerotype:
            x_plus = torch.ones(x.shape[0], 1)
        else:
            x_plus = torch.cat([x, torch.ones(x.shape[0], 1)], dim=1)  # todo
        # Shape of weighted_x is n_cases * n_rules * (n_in+1)
        weights[torch.isnan(weights)] = 1e-9
        weighted_x = torch.einsum('bp, bq -> bpq', weights, x_plus)
        # Can't have value 0 for weights, or LSE won't work:
        # print(True in weighted_x == 0)
        # if (True in weighted_x == 0):
        #     exit(0)
        # print(weighted_x.shape)

        weighted_x[weighted_x == 0] = 1e-12
        # Squash x and y down to 2D matrices for gels:
        weighted_x_2d = weighted_x.view(weighted_x.shape[0], -1)
        y_actual_2d = y_actual.view(y_actual.shape[0], -1)
        # Use gels to do LSE, then pick out the solution rows:
        try:
            # coeff_2d, _ = torch.gels(y_actual_2d, weighted_x_2d)
            coeff_2d, _, _, _ = torch.linalg.lstsq(weighted_x_2d, y_actual_2d)
            # coeff_2d, _, _, _ = torch.linalg.lstsq(y_actual_2d, weighted_x_2d)
        except RuntimeError as e:
            print('Internal error in gels', e)
            print('weights:', weights)
            print('plus:', x_plus)
            print('Weights are:', weighted_x)
            raise e
        coeff_2d = coeff_2d[0:weighted_x_2d.shape[1]]
        # print(coeff_2d.shape)
        # exit(0)
        # Reshape to 3D tensor: divide by rules, n_in+1, then swap last 2 dims
        if self.zerotype:
            self.coeff = coeff_2d.reshape(weights.shape[1], 1, -1) \
                .transpose(1, 2)
        else:
            self.coeff = coeff_2d.reshape(weights.shape[1], x.shape[1] + 1, -1) \
                .transpose(1, 2)
        # coeff dim is thus: n_rules * n_out * (n_in+1)
        # print(self.coeff)
        # exit(0)

    def forward(self, x):
        '''
            Calculate: y = coeff * x + const   [NB: no weights yet]
                  x.shape: n_cases * n_in
              coeff.shape: n_rules * n_out * (n_in+1)
                  y.shape: n_cases * n_out * n_rules
        '''
        # print(x.shape)
        # Append 1 to each list of input vals, for the constant term:
        if self.zerotype:
            x_plus = torch.ones(x.shape[0], 1)
        else:
            x_plus = torch.cat([x, torch.ones(x.shape[0], 1)], dim=1)
        # Need to switch dimension for the multipy, then switch back:
        # print(self._coeff)
        # print(x_plus.shape)
        # exit(0)
        y_pred = torch.matmul(self.coeff, x_plus.t())
        # y_pred[torch.isnan(y_pred)] = 1e-9
        # print(y_pred.shape)
        # exit(0)
        return y_pred.transpose(0, 2)  # swaps cases and rules


class PlainConsequentLayer(ConsequentLayer):
    '''
        A linear layer to represent the TSK consequents.
        Not hybrid learning, so coefficients are backprop-learnable parameters.
    '''
    def __init__(self, *params):
        super(PlainConsequentLayer, self).__init__(*params)
        self.register_parameter('coefficients',
                                torch.nn.Parameter(self._coeff))

    @property
    def coeff(self):
        '''
            Record the (current) coefficients for all the rules
            coeff.shape: n_rules * n_out * (n_in+1)
        '''
        return self.coefficients

    def fit_coeff(self, x, weights, y_actual):
        '''
        '''
        assert False,\
            'Not hybrid learning: I\'m using BP to learn coefficients'


class WeightedSumLayer(torch.nn.Module):
    '''
        Sum the TSK for each outvar over rules, weighted by fire strengths.
        This could/should be layer 5 of the Anfis net.
        I don't actually use this class, since it's just one line of code.
    '''
    def __init__(self):
        super(WeightedSumLayer, self).__init__()

    def forward(self, weights, tsk):
        '''
            weights.shape: n_cases * n_rules
                tsk.shape: n_cases * n_out * n_rules
             y_pred.shape: n_cases * n_out
        '''
        # Add a dimension to weights to get the bmm to work:
        y_pred = torch.bmm(tsk, weights.unsqueeze(2))
        return y_pred.squeeze(2)


class scikit_anfis(torch.nn.Module, BaseEstimator, TransformerMixin):
    '''
        This is a container for the 5 layers of the ANFIS net.
        The forward pass maps inputs to outputs based on current settings,
        and then fit_coeff will adjust the TSK coeff using LSE.
    '''
    def __init__(self, fs=None, data=None, outvarnames=['y0'], epoch=10, description=None, rules=None, label="r", hybrid=True, zerotype=False, show_banner=False):
        super(scikit_anfis, self).__init__()
        self.fs = fs
        self.rules = rules
        self.invardefs = data
        self.zerotype = zerotype
        # print(self.fs.coeff)

        if self.fs is None:
            # print(type(invardefs))
            if isinstance(self.invardefs, np.ndarray):
                new_invardefs = [('x'+str(i), make_gauss_mfs(2, [4.3, 7.9])) for i in range(self.invardefs.shape[1])]
            else:
                new_invardefs = self.invardefs
        else:
            # self.coeff = self.fs.coeff
            new_invardefs = self.fs.antec
            # print(new_invardefs)
            if rules is None:
                self.rules = self.fs._rules
        # print(self.invardefs)

        self.description = description
        self.outvarnames = outvarnames
        self.hybrid = hybrid
        self.is_training = False
        self.epoch = epoch
        self.label = label
        self.show_banner = show_banner



        # self.num_in = 0
        # self.num_rules = 0

        if not self.rules:
            # print(self.invardefs)
            if isinstance(new_invardefs, list):
                varnames = [v for v, _ in new_invardefs]
                mfdefs = [FuzzifyVariable(mfs) for _, mfs in new_invardefs]
            else:
                varnames = [v for v, _ in new_invardefs.items()]
            # print(varnames)
                mfdefs = [FuzzifyVariable(mfs) for _, mfs in new_invardefs.items()]
        else:
            varnames = [v for v in new_invardefs.keys()]
            mfdefs = [FuzzifyVariable(mfs) for mfs in new_invardefs.values()]

        self.num_in = len(new_invardefs)
        if not self.rules:
            self.num_rules = np.prod([len(mfs) for _, mfs in new_invardefs])
        else:
            self.num_rules = len(self.rules)
        # print(varnames)
        # print(mfdefs)
        if self.hybrid:
            cl = ConsequentLayer(self.num_in, self.num_rules, self.num_out, self.zerotype)
        else:
            cl = PlainConsequentLayer(self.num_in, self.num_rules, self.num_out, self.zerotype)
        # print('haha2')
        self.layer = torch.nn.ModuleDict(OrderedDict([
            ('fuzzify', FuzzifyLayer(mfdefs, varnames)),
            ('rules', AntecedentLayer(mfdefs) if not self.rules else AntecedentLayer(mfdefs, self.rules, varnames)), #
            # normalisation layer is just implemented as a function.
            ('consequent', cl),
            # weighted-sum layer is just implemented as a function.
            ]))

        # print(self.fs.coeff)

        if self.fs is not None:
            self.coeff = self.fs.coeff

        if self.show_banner:
            self._banner()

    def _banner(self):
        # import pkg_resources
        vrs = '2.1.3' # pkg_resources.get_distribution('scikit_anfis').version
        print(" ____   __  __  __  ____    ___  __  ____  __  ____")
        print("/ ___) (  ) \\ \\/ / / /\\ \\  ( _ \\(  )(  __)(  )/ ___) v%s " % vrs)
        print(" \\___ \\ )(   )  ( / /__\\ \\  )(\\ \\)(  ) _)  )(  \\___ \\ ")
        print("(____/ (__)  (__)/_/    \\_\\(__)\\ __)(__)  (__)(____/")
        print()
        print(" Created by Dongsong Zhang (dsongzhang@gmail.com)")
        print(" and Tianhua Chen (T.Chen@hud.ac.uk) in 2023")
        print()

    @property
    def num_out(self):
        return len(self.outvarnames)

    @property
    def coeff(self):
        return self.layer['consequent'].coeff

    # @coeff.setter
    # def invardefs(self, new_invardefs):
    #     self.invardefs = new_invardefs

    @coeff.setter
    def coeff(self, new_coeff):
        # print(self.fs.coeff)
        # print(new_coeff)
        # exit(0)
        if isinstance(new_coeff, np.ndarray):
            new_coeff = torch.tensor(new_coeff).float().unsqueeze(1)
        if self.layer['consequent'].coeff.nelement() == 0:
            self.layer['consequent'].coeff = new_coeff
        else:
            # self.layer['consequent'].coeff.data.copy_(new_coeff.data)
            if not isinstance(new_coeff, type(None)):
                self.layer['consequent'].coeff.data = new_coeff.data.clone()
        # print(self.layer['consequent'].coeff)
        # print(new_coeff)
        # exit(0)

    @property
    def antec(self):
        self.load()
        return self.layer['fuzzify'].antec

    def fit_coeff(self, x, y_actual):
        '''
            Do a forward pass (to get weights), then fit to y_actual.
            Does nothing for a non-hybrid ANFIS, so we have same interface.
        '''
        # print('into hybrid')
        # print(self.hybrid)
        if self.hybrid:
            # self(x)
            self.layer['consequent'].fit_coeff(x, self.weights, y_actual)

    def input_variables(self):
        '''
            Return an iterator over this system's input variables.
            Yields tuples of the form (var-name, FuzzifyVariable-object)
        '''
        return self.layer['fuzzify'].varmfs.items()

    def output_variables(self):
        '''
            Return an list of the names of the system's output variables.
        '''
        return self.outvarnames

    def extra_repr(self):
        rstr = []
        vardefs = self.layer['fuzzify'].varmfs
        rule_ants = self.layer['rules'].extra_repr(vardefs).split('\n')
        for i, crow in enumerate(self.layer['consequent'].coeff):
            rstr.append('Rule {:2d}: IF {}'.format(i, rule_ants[i]))
            rstr.append(' '*9+'THEN {}'.format(crow.tolist()))
        return '\n'.join(rstr)

    def print_rules(self):
        # print(self.rules)
        rstr = []
        # vardefs = self.layer['fuzzify'].varmfs
        # rule_ants = self.layer['rules'].extra_repr(vardefs).split('\n')
        # for i, crow in enumerate(self.layer['consequent'].coeff):
        #     # rstr.append('Rule {:2d}: IF {}'.format(i, rule_ants[i]))
        #     # rstr.append(' ' * 9 + 'THEN {} is {}'.format(self.outvarnames[0], crow.tolist()))
        #     rstr.append('Rule {:2d}: IF {} THEN {} is {}'.format(i+1, rule_ants[i], self.rules[i][1][0], self.rules[i][1][1]))

        i = 1
        for item in self.rules:
            rule = "rule " + str(i) + ": IF "
            # print(item)
            ps = str(item[0])
            ps = re.sub("AND_p", "and", ps)
            ps = re.sub("IS", "is", ps)
            ps = re.sub("f\.\(", "", ps)
            ps = re.sub("\)\)", ")", ps)
            ps = re.sub("c.", "", ps)
            rule += ps
            rule += " THEN "
            rule += f"({item[1][0]} is {item[1][1]})"
            rstr.append(rule)
            i += 1
        print('\n'.join(rstr))

    def fit(self, X, y=""):
        # print('scikit_anfis fit')
        # print(X)
        if isinstance(X, pandas.DataFrame):
            X = X.to_numpy()
        if len(y) == 0 or not y.any():
            X = torch.from_numpy(X).float()
            X = DataLoader(TensorDataset(X[:, 0:-1], torch.unsqueeze(X[:, -1], -1)), batch_size=1024, shuffle=True)
        else:
            X = torch.from_numpy(X).float()
            y = torch.from_numpy(y).float()
            X = DataLoader(TensorDataset(X, torch.unsqueeze(y, -1)), batch_size=1024, shuffle=True)
        train_anfis(self, X, self.epoch)
        # exit(0)
        return self

    def transform(self, X, y=None):
        # print('scikit_anfis transform')
        return X.dataset.tensors[0].numpy(), self.antec, torch.squeeze(self.coeff).cpu().data.numpy()

    def predict(self, X, y=None):
        # print('scikit_anfis predict')
        # print(type(X))
        # exit(0)
        if isinstance(X, pandas.DataFrame):
            X = X.to_numpy()
        X = torch.from_numpy(X).float()
        # X = DataLoader(TensorDataset(X), batch_size=1024, shuffle=True)
        X = DataLoader(X, batch_size=1024, shuffle=True)
        y_pred = test_anfis(self, X).detach().numpy()
        # y_pred = y_pred.detach().numpy()
        np.nan_to_num(y_pred, copy=False)
        if self.label == "c":
            y_pred = np.round(np.abs(y_pred))
        return y_pred

    def predict_proba(self, X, y=None):
        # print('scikit_anfis predict probably')
        # if self.label == "r":
        #     raise ValueError("predict_proba can only be used when label=\"c\"")
        y_pred = self.predict(X)
        return softmax(y_pred, axis=1)


    def save(self, path='tmp.pkl'):
        """

        Save model.

        :param str path: Model save path.
        """
        # torch.save(self.state_dict(), path)
        # print('state_dict:', self.state_dict())
        # print('coeff:', self.coeff)
        torch.save({
            # 'epoch': epoch,
            'coeff': self.coeff,
            'model_state_dict': self.state_dict(),
            # 'optimizer_state_dict': optimizer.state_dict(),
            # 'loss': loss,
        }, path)

    def load(self, path='tmp.pkl'):
        """

        Load model.

        :param str path: Model save path.
        """
        # self.load_state_dict(torch.load(path))
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # print(checkpoint['coeff'])
        # print(self.coeff)
        # exit(0)
        if self.hybrid:
            self.coeff = checkpoint['coeff']
        # epoch = checkpoint['epoch']
        # loss = checkpoint['loss']

    def forward(self, x, y_actual=None):
        '''
            Forward pass: run x thru the five layers and return the y values.
            I save the outputs from each layer to an instance variable,
            as this might be useful for comprehension/debugging.
        '''
        # print('x:',x)
        self.fuzzified = self.layer['fuzzify'](x)
        # print('fuzzified:', self.fuzzified.shape)
        self.raw_weights = self.layer['rules'](self.fuzzified)
        # print('raw_weights:', self.raw_weights)

        self.weights = F.normalize(self.raw_weights, p=1, dim=1)
        # print(self.layer['fuzzify'].__repr__)
        # print('weights:', self.weights.shape)
        # exit(0)
        if self.is_training:
            self.fit_coeff(x, y_actual)
        self.rule_tsk = self.layer['consequent'](x)
        # print(self.rule_tsk.shape)
        # exit(0)
        # y_pred = self.layer['weighted_sum'](self.weights, self.rule_tsk)
        y_pred = torch.bmm(self.rule_tsk, self.weights.unsqueeze(2))
        self.y_pred = y_pred.squeeze(2)
        # print(self.y_pred)
        # exit(0)
        return self.y_pred


# These hooks are handy for debugging:

def module_hook(label):
    ''' Use this module hook like this:
        m = scikit_anfis()
        m.layer.fuzzify.register_backward_hook(module_hook('fuzzify'))
        m.layer.consequent.register_backward_hook(modul_hook('consequent'))
    '''
    return (lambda module, grad_input, grad_output:
            print('BP for module', label,
                  'with out grad:', grad_output,
                  'and in grad:', grad_input))


def tensor_hook(label):
    '''
        If you want something more fine-graned, attach this to a tensor.
    '''
    return (lambda grad:
            print('BP for', label, 'with grad:', grad))
