# -*- coding: utf-8 -*-
import matplotlib as mpl
mpl.use("pgf")
import numpy as np
import plot_functions as pf
import os
import csv
import math
import matplotlib.pyplot as pyplot
import pylab as P
import matplotlib
matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = True
# frequency in Hz, gain in dB, phase in radians between -pi and pi. They should
# all be vectors of the same length.


def plot_exemplary_roc(fname=None):
    if (fname is not None and mpl.get_backend() == u'pgf'):
        fileName, fileExtension = os.path.splitext(fname)
        if (fileExtension == "pdf"):
            pf.latexify()
        if(fileExtension == "pgf"):
            pf.latexifypgf()
    frange = [0.0, 1.0]
    magrange = [0.0, 1.0]
    fig, ax1 = pyplot.subplots()
    pf.format_axes(ax1)
    colors = pf.getColorList(7)
    pyplot.subplots_adjust(
        left=0.125, bottom=0.125, right=0.92, top=0.9, wspace=0.2, hspace=0.3)

    # add title, if given
    # pyplot.title("BI measurement with denoising ");
    npzfile = np.load(pf.data_dir() + 'exampleROC.npz')
    fpr = npzfile['fpr']
    tpr = npzfile['tpr']
    mean_fpr = npzfile['mean_fpr']
    pyplot.plot(fpr, tpr, color=colors[
                1], linestyle='-', linewidth=1.5, label="trained classifier")

    pyplot.plot(mean_fpr, mean_fpr, color=colors[5], linestyle='--', linewidth=1.5, label="random classifier")

    pf.modifyLegend(pyplot.legend(loc=4))
    # plot it as a log-scaled graph
    # update axis ranges
    ax_lim = []
    ax_lim[0:4] = pyplot.axis()
    # check if we were given a frequency range for the plot
    if (frange != None):
        ax_lim[0:2] = frange
    # check if we were given a dB range for the magnitude part of the plot
    # magrange = [0.2, 1.01]
    if (magrange != None):
        ax_lim[2:4] = magrange

    pyplot.axis(ax_lim)

    pyplot.grid(True)
    # turn on the minor gridlines to give that awesome log-scaled look
    pyplot.grid(True, which='minor')
    pyplot.xlabel("False positive rate")
    # pyplot.title("(b)");
    pyplot.ylabel("True positive rate")
    if (fname != None and mpl.get_backend() == u'pgf'):
        pyplot.savefig(fname, dpi=pf.getDpi())
    else:
        pyplot.show()



if __name__ == '__main__':

    plot_exemplary_roc('../5/pgf/Exemplary_ROC.pgf')

