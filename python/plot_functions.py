# -*- coding: utf-8 -*-
#import matplotlib as mpl
#mpl.use("pgf")
import numpy as np
from math import sqrt
import pandas as pd
SPINE_COLOR = 'gray'
import csv
import math
import matplotlib.pyplot as pyplot
import pylab as P
import matplotlib
import brewer2mpl

def getDpi():
        return 600

def getRasterized():
        return False
def value_for_window_min_max(data, start, stop,meanValue):
        minValue = data[start]
        maxValue = data[start]

        for i in range(start,stop):
                if data[i] < minValue:
                        minValue = data[i]
                if data[i] > maxValue:
                        maxValue = data[i]

        if abs(minValue-meanValue) > abs(maxValue-meanValue):
                return minValue
        else:
                return maxValue

# This will only work properly if window_size divides evenly into len(data)
def subsample_data(data, window_size):


        out_data = []
        data = data[:len(data)-len(data)%window_size]
        print(len(data))
        print(len(data)/window_size)
        meanValue = np.mean(data)

        for i in range(0,int((len(data)/window_size))):
                out_data.append(value_for_window_min_max(data,i*window_size,i*window_size+window_size-1,0))

        return out_data
def modifyLegend(legend):
        frame = legend.get_frame()
        legend.shadow = False
        legend.fancybox=True
        #for label in legend.get_lines():
        #	label.set_linewidth(0.5)  # the legend line width
        for label in legend.get_texts():
                label.set_fontsize(10)
        frame.set_facecolor('0.99')
        frame.set_linewidth(0.5)
        frame.set_alpha(1.0)
        #frame.set_edgecolor('1.0')
        return legend

def getColorList(number):
        # brewer2mpl.get_map args: set name  set type  number of colors
        #['Accent', 'Dark2', 'Paired', 'Pastel1', 'Pastel2', 'Set1', 'Set2', 'Set3']
        bmap = brewer2mpl.get_map('Paired', 'qualitative', number)
        colors = bmap.mpl_colors
        return colors

def data_dir():
        return "../Data/"

def figsize(scale):
        fig_width_pt = 469.755 # Get this from LaTeX using \the\textwidth
        inches_per_pt = 1.0/72.27 # Convert pt to inch
        golden_mean = (np.sqrt(5.0)-1.0)/2.0 # Aesthetic ratio (you could change this)
        fig_width = fig_width_pt*inches_per_pt*scale # width in inches
        fig_height = fig_width*golden_mean # height in inches
        fig_size = [fig_width,fig_height]
        return fig_size

def latexifypgf():
        """Set up matplotlib's RC params for LaTeX pdf plotting.
        Call this before plotting a figure.
        """

        pgf_with_latex = { # setup matplotlib to use latex for output
                           "pgf.texsystem": "pdflatex", # change this if using xetex or lautex
                           "figure.autolayout": True,
                           "text.usetex": True, # use LaTeX to write all text
                           "font.family": "serif",
                           "font.serif": [], # blank entries should cause plots to inherit fonts from the document
                           "font.sans-serif": [],
                           "font.monospace": [],
                           "axes.labelsize": 10, # LaTeX default is 10pt font.
                           "font.size": 10,
                           "legend.fontsize": 8, # Make the legend/label fonts a little smaller
                           "xtick.labelsize": 8,
                           "ytick.labelsize": 8,
                           "figure.figsize": figsize(0.9), # default fig size of 0.9 textwidth
                           "pgf.preamble": [
                                   r"\usepackage[utf8x]{inputenc}", # use utf8 fonts becasue your computer can handle it :)
                                   r"\usepackage{libertine}",
                                   r"\usepackage[libertine]{newtxmath}",
                                   r"\usepackage[T1]{fontenc}", # plots will be generated using this preamble
                           ]
                           }

        matplotlib.rcParams.update(pgf_with_latex)


def latexify(fig_width=None, fig_height=None, columns=1):
        """Set up matplotlib's RC params for LaTeX plotting.
        Call this before plotting a figure.

        Parameters
        ----------
        fig_width : float, optional, inches
        fig_height : float,  optional, inches
        columns : {1, 2}
        """

        # code adapted from http://www.scipy.org/Cookbook/Matplotlib/LaTeX_Examples

        # Width and max height in inches for IEEE journals taken from
        # computer.org/cms/Computer.org/Journal%20templates/transactions_art_guide.pdf

        assert(columns in [1,2])

        if fig_width is None:
                fig_width = 3.39 if columns==1 else 6.9 # width in inches

        if fig_height is None:
                golden_mean = (sqrt(5)-1.0)/2.0    # Aesthetic ratio
                fig_height = fig_width*golden_mean # height in inches

        MAX_HEIGHT_INCHES = 8.0
        if fig_height > MAX_HEIGHT_INCHES:
                print("WARNING: fig_height too large:" + fig_height + 
                      "so will reduce to" + MAX_HEIGHT_INCHES + "inches.")
                fig_height = MAX_HEIGHT_INCHES

        params = {"backend": "ps",
                  "text.usetex": True, # use LaTeX to write all text
                  "font.family": "serif",
                  #"font.serif": [], # blank entries should cause plots to inherit fonts from the document
                  #"font.sans-serif": [],
                  #"font.monospace": [],
                  "axes.labelsize": 10, # LaTeX default is 10pt font.
                  "font.size": 10,
                  "legend.fontsize": 8, # Make the legend/label fonts a little smaller
                  "xtick.labelsize": 8,
                  "ytick.labelsize": 8,
                  "figure.figsize": figsize(0.9), # default fig size of 0.9 textwidth
                  "text.latex.preamble": [
                          r"\usepackage[utf8x]{inputenc}", # use utf8 fonts becasue your computer can handle it :)
                          r"\usepackage{libertine}",
                          r"\usepackage[libertine]{newtxmath}",
                          r"\usepackage[T1]{fontenc}", # plots will be generated using this preamble
                  ]
                  }

        matplotlib.rcParams.update(params)



def format_axes(ax):

        for spine in ['top', 'right']:
                ax.spines[spine].set_visible(False)

        for spine in ['left', 'bottom']:
                ax.spines[spine].set_color(SPINE_COLOR)
                ax.spines[spine].set_linewidth(1.0)

        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

        for axis in [ax.xaxis, ax.yaxis]:
                axis.set_tick_params(direction='out', color=SPINE_COLOR)

        return ax

