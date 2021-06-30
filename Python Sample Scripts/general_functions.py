#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 20 10:57:53 2021

@author: shantanu
"""
import time
import numpy as np
import matplotlib.pyplot as plt

def TicTocGenerator():
    # Generator that returns time differences
    ti = 0           # initial time
    tf = time.time() # final time
    while True:
        ti = tf
        tf = time.time()
        yield tf-ti # returns the time difference
TicToc = TicTocGenerator() # create an instance of the TicTocGen generator
# This will be the main function through which we define both tic() and toc()
def toc(tempBool=True):
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicToc)
    if tempBool:
        print( "\nElapsed time: %f seconds.\n" %tempTimeInterval )
def tic():
    # Records a time in TicToc, marks the beginning of a time interval
    toc(False)

def bestfit_line_plot(X, Y, labels, marker_line_type=['ok', 'Xr', 'Pb', 'Dg'], legend_loc='best', markersize=10, linewidth=2, xlabel='$\Delta$ x', ylabel='Error', title=None, plot_flag=True):
    assert len(X)==Y.shape[1]
    x_fit=np.tile(X, (Y.shape[0], 1)).ravel(); y_fit=Y.ravel();
    mc=np.polyfit(np.log(x_fit), np.log(y_fit), 1);
    dx_fit=np.linspace(np.min(X), np.max(X), num=50); error_fit=np.exp(mc[0]*np.log(dx_fit) + mc[1]);
    if plot_flag:
        plt.figure();
        for i in range(Y.shape[0]):
            plt.loglog(X, Y[i,:], marker_line_type[i], markersize=markersize, linewidth=linewidth, label=labels[i]);
        plt.loglog(dx_fit, error_fit, '-r', markersize=markersize, linewidth=linewidth, label='Slope: '+'{0:.2f}'.format(mc[0]));
        plt.legend(loc=legend_loc); plt.xlabel(xlabel); plt.ylabel(ylabel);
        plt.grid(b=True, which='major', color='k', linestyle='-');
        plt.grid(b=True, which='minor', color='k', linestyle='--'); plt.title(title)
    return mc[0]