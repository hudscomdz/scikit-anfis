#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    ANFIS in torch: some simple functions to supply data and plot results.
    @author: James Power <james.power@mu.ie> Apr 12 18:13:10 2019
"""

import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import numpy as np
# import bcolors

dtype = torch.float

windows = []

def CalcCorrelation(a, b):
    """
    Calculates the correlation between arrays <a> and <b>. If the arrays are
    multi-column, the correlation is calculated as all the columns are one
    single vector.
    """
    # Convert to (n, ) dimension
    a = a.flatten()
    b = b.flatten()

    # Correlation
    corr = np.corrcoef(a, b)[0, 1]

    return corr


def CalcAccuracy(a, b):
    """
    Calculates the accuracy (in %) between arrays <a> and <b>. The two arrays
    must be column/row vectors.
    """
    # Convert to (n, ) dimension
    a = a.flatten()
    b = b.flatten()

    # Correlation
    accu = 100.0 * (a == b).sum() / len(a)

    return accu


class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, x, y):
        # criterion = torch.nn.MSELoss(reduction='sum')
        criterion = torch.nn.MSELoss()
        loss = torch.sqrt(criterion(x, y))
        # eps = 1e-5
        # loss = torch.sqrt(criterion(x, y) + eps)
        # # loss = torch.sqrt(criterion(x, y) * x.size(0) + eps)
        return loss

class TwoLayerNet(torch.nn.Module):
    '''
        From the pytorch examples, a simple 2-layer neural net.
        https://pytorch.org/tutorials/beginner/pytorch_with_examples.html
    '''
    def __init__(self, d_in, hidden_size, d_out):
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(d_in, hidden_size)
        self.linear2 = torch.nn.Linear(hidden_size, d_out)

    def forward(self, x):
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred


def linear_model(x, y, epochs=200, hidden_size=10):
    '''
        Predict y from x using a simple linear model with one hidden layer.
        https://pytorch.org/tutorials/beginner/pytorch_with_examples.html
    '''
    assert x.shape[0] == y.shape[0], 'x and y have different batch sizes'
    d_in = x.shape[1]
    d_out = y.shape[1]
    model = TwoLayerNet(d_in, hidden_size, d_out)
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    errors = []
    for t in range(epochs):
        y_pred = model(x)
        tot_loss = criterion(y_pred, y)
        perc_loss = 100. * torch.sqrt(tot_loss).item() / y.sum()
        errors.append(perc_loss)
        if t % 10 == 0 or epochs < 20:
            print('epoch {:4d}: {:.5f} {:.2f}%'.format(t, tot_loss, perc_loss))
        optimizer.zero_grad()
        tot_loss.backward()
        optimizer.step()
    return model, errors


def plotErrors(errors):
    '''
        Plot the given list of error rates against no. of epochs
    '''
    plt.plot(range(len(errors)), errors, '-ro', label='errors')
    plt.ylabel('root mean squares error')#'Percentage error')
    plt.xlabel('Epoch')
    plt.show()


def plotResults(y_actual, y_predicted):
    '''
        Plot the actual and predicted y values (in different colours).
    '''
    plt.plot(range(len(y_predicted)), y_predicted.detach().numpy(),
             'r', label='trained')
    plt.plot(range(len(y_actual)), y_actual.numpy(), 'b', label='original')
    plt.legend(loc='upper left')
    plt.show()


def _plot_mfs(var_name, fv, x):
    '''
        A simple utility function to plot the MFs for a variable.
        Supply the variable name, MFs and a set of x values to plot.
    '''
    # Sort x so we only plot each x-value once:
    xsort, _ = x.sort()
    for mfname, yvals in fv.fuzzify(xsort):
        plt.plot(xsort.tolist(), yvals.tolist(), label=mfname)
    plt.xlabel('Values for variable {} ({} MFs)'.format(var_name, fv.num_mfs))
    plt.ylabel('Membership')
    plt.legend(bbox_to_anchor=(1., 0.95))
    plt.show()


def plot_all_mfs(model, x):
    for i, (var_name, fv) in enumerate(model.layer.fuzzify.varmfs.items()):
        _plot_mfs(var_name, fv, x[:, i])


def calc_error(y_pred, y_actual):
    with torch.no_grad():
        tot_loss = F.mse_loss(y_pred, y_actual)
        rmse = torch.sqrt(tot_loss).item()
        # print(y_pred)
        # print(y_actual)
        # perc_loss = torch.mean(100. * torch.abs((y_pred - y_actual)
        #                        / y_actual))
    # return (tot_loss, rmse, perc_loss)
    return (tot_loss, rmse)


def test_anfis(model, data, show_plots=False):
    '''
        Do a single forward pass with x and compare with y_actual.
    '''
    # x, y_actual = data.dataset.tensors
    x = data.dataset
    # print(data.__dict__)
    # print(x.shape)
    # exit(0)
    if show_plots:
        plot_all_mfs(model, x)
    # print('### Testing for {} cases'.format(x.shape[0]))
    # print(model.layer['fuzzify'].__repr__)
    # print(model.coeff.shape)
    model.load()
    # print(model.layer['fuzzify'].__repr__)
    # print(model.coeff.shape)
    # exit(0)
    model.eval()
    model.is_training = False
    y_pred = model(x)#, y_actual)
    # print(type(y_pred))
    # exit(0)
    # print('input Mfs parameters:')
    # print(model.layer['fuzzify'].antec)
    # print('output parameters:')
    # print(model.coeff.data)
    # print(model.layer['fuzzify'].__repr__)
    # print(model.layer['fuzzify'].__antec__)
    # print('y_pred:', y_pred)
    # mse, rmse, perc_loss = calc_error(y_pred, y_actual)
    # print('MS error={:.5f}, RMS error={:.5f}, percentage={:.2f}%'
    #       .format(mse, rmse, perc_loss))

    # mse, rmse = calc_error(y_pred, y_actual)
    # print('MS Error={:.5f}, RMS Error={:.5f}'.format(mse, rmse))
    # if show_plots:
    #     plotResults(y_actual, y_pred)
    return y_pred


def customLR(epoch, errors, normg):
    k = 1
    global windows
    # print(epoch)
    # exit(0)
    if len(errors) <= 4:
        return k

    if len(windows) == 0:
        windows.append(errors[-3])
        windows.append(errors[-2])
        windows.append(errors[-1])
        # windows = errors[-4:-1]
        # print(epoch, windows)
    else:
        del windows[0]
        return k

    if (errors[-4] > errors[-3] > errors[-2] > errors[-1]):
        k = 1.1

    if (errors[-4] > errors[-5]) and (errors[-4] > errors[-3]) and (errors[-2] > errors[-3]) and (errors[-2] > errors[-1]):
        k = 0.9

    # print(epoch, k, windows, errors[-5:-1])
    # exit(0)
    # kk = k*154.16
    # print('kk=', kk)
    return k*normg


def train_anfis_with(model, data, optimizer, criterion, epochs=500, show_plots=False):
    '''
        Train the given model using the given (x,y) data.
    '''
    # torch.autograd.set_detect_anomaly(True)
    # #torch.autograd.detect_anomaly(True)
    model.is_training = True
    errors = []  # Keep a list of these for plotting afterwards
    min_training_RMSE = float('inf')
    normg = 1.0
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    lmbda = lambda epoch: customLR(epoch, errors, normg)
    scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmbda)
    # print('### Training for {} epochs, training size = {} cases'.
    #       format(epochs, data.dataset.tensors[0].shape[0]))

    for t in range(epochs):
        # Process each mini-batch in turn:
        x, y_actual = data.dataset.tensors
        # print(x.size())
        # exit(0)
        # print(model.layer['fuzzify'].__repr__)
        y_pred = model(x, y_actual)
        # mse, rmse, perc_loss = calc_error(y_pred, y_actual)
        mse, rmse = calc_error(y_pred, y_actual)
        # errors.append(perc_loss)
        errors.append(rmse)
        if rmse < min_training_RMSE:
            # print('rmse:', rmse)
            min_training_RMSE = rmse
            model.save()

        optimizer.zero_grad()
        loss = criterion(y_pred, y_actual)
        # print('loss:', loss.data)
        # print(model.layer['fuzzify'].__repr__)

        loss.backward()
        optimizer.step()
        scheduler.step()

        # print('epoch=', t + 1, ' lr=', optimizer.param_groups[0]["lr"])

        # print('MFs parameters:', model.layer['fuzzify'].__repr__)
        # print('coeff:', model.coeff)

        # print('epoch {:4d}: MSE={:.5f}, RMSE={:.5f}'
        #       .format(t + 1, mse, rmse))
    # End of training, so graph the results:
    # print('Minimal training RMSE = {:.5f}'.format(min_training_RMSE))
    # print('y_pred[0]:', y_pred[0])
    # print(model.coeff)
    # exit(0)

    if show_plots:
        # print(errors)
        # exit(0)
        plotErrors(errors)
        # y_actual = data.dataset.tensors[1]
        # y_pred = model(data.dataset.tensors[0])
        # plotResults(y_actual, y_pred)


def train_anfis(model, data, epochs=10, show_plots=False):
    '''
        Train the given model using the given (x,y) data.
    '''
    # print('haha')
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.99)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01)#*153.846)
    # optimizer = torch.optim.Rprop(model.parameters(), lr=0.01)
    # optimizer = torch.optim.LBFGS(model.parameters(), lr=0.01, line_search_fn='strong_wolfe')
    # optimizer = torch.optim.Adadelta(model.parameters(), lr=0.01)
    # optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01)
    # criterion = torch.nn.MSELoss(reduction='sum')
    criterion = RMSELoss()
    train_anfis_with(model, data, optimizer, criterion, epochs, show_plots)


def train_test_anfis_with(model, data, test_data, optimizer, criterion,
                     epochs=500, show_plots=False):
    '''
        Train the given model using the given (x,y) data.
    '''
    errors = []  # Keep a list of these for plotting afterwards
    min_training_RMSE = 1.0
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # print('### Training for {} epochs, training size = {} cases'.
    #       format(epochs, data.dataset.tensors[0].shape[0]))
    for t in range(epochs):
        # Process each mini-batch in turn:
        for x, y_actual in data:
            # print(x.shape)
            # print(y_actual.shape)
            # print(torch.sum(y_actual))
            y_pred = model(x)
            # print(y_pred)
            # Compute and print loss
            loss = criterion(y_pred, y_actual)
            # print(loss)
            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # Epoch ending, so now fit the coefficients based on all data:
        x, y_actual = data.dataset.tensors
        # print('training end')
        # print(model.coeff.grad)
        # print(model.coeff.data)
        # print(model.coeff)
        # exit(0)
        with torch.no_grad():
            # print('into no_grad:')
            model.fit_coeff(x, y_actual)
        # Get the error rate for the whole batch:
        y_pred = model(x)
        # mse, rmse, perc_loss = calc_error(y_pred, y_actual)
        mse, rmse = calc_error(y_pred, y_actual)
        # errors.append(perc_loss)
        errors.append(rmse)
        if rmse < min_training_RMSE:
            min_training_RMSE = rmse
        # Print some progress information as the net is trained:
        # if epochs < 30 or t % 10 == 0:
        #     print('epoch {:4d}: MSE={:.5f}, RMSE={:.5f}, percentage={:.2f}%'
        #           .format(t+1, mse, rmse, perc_loss))

        # print('epoch {:4d}: MSE={:.5f}, RMSE={:.5f}, percentage={:.2f}%'
        #       .format(t + 1, mse, rmse, perc_loss))
        print('epoch {:4d}: MSE={:.5f}, RMSE={:.5f}'
              .format(t + 1, mse, rmse))



    # End of training, so graph the results:
    print('Minimal training RMSE = {:.5f}'.format(min_training_RMSE))
    if show_plots:
        plotErrors(errors)
        # y_actual = data.dataset.tensors[1]
        # y_pred = model(data.dataset.tensors[0])
        # plotResults(y_actual, y_pred)


def train_test_anfis(model, data, testdata, epochs=500, show_plots=False):
    '''
        Train the given model using the given (x,y) data.
    '''
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.99)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss(reduction='sum')
    train_test_anfis_with(model, data, testdata, optimizer, criterion, epochs, show_plots)

if __name__ == '__main__':
    x = torch.arange(1, 100, dtype=dtype).unsqueeze(1)
    y = torch.pow(x, 3)
    model, errors = linear_model(x, y, 100)
    plotErrors(errors)
    plotResults(y, model(x))
