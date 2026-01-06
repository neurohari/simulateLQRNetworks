import matplotlib.pyplot as plt
import matplotlib
import numpy as np

def load_plot_setting():
    # Configure figure
    #plt.style.use('seaborn-paper')
    matplotlib.rc('font',**{'family':'sans-serif'})
    SMALL_SIZE = 22 
    MEDIUM_SIZE = 22 
    BIGGER_SIZE = 22 
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    plt.rc('xtick', direction='out')
    plt.rc('ytick', direction='out')
    plt.rc('xtick', top=False)
    plt.rc('ytick', right=False)
    plt.rc('axes.spines', right=False)
    plt.rc('axes.spines', top=False)


def plot_eig(W_evals, label="$W$", figsize=(5,5)):
    plt.figure(figsize=figsize)
    plt.plot(W_evals.real,W_evals.imag,'o',color='grey')
    plt.axis([-10,2,-5,5])
    #plt.xticks(np.arange(-10, 12, step=5))
    #plt.yticks(np.arange(-10, 12, step=5))
    plt.ylabel("imag. $\lambda$")
    plt.xlabel("real $\lambda$")
    plt.text(3,8,"unstable",fontsize=22)
    plt.text(-1,8,"stable",fontsize=22,horizontalalignment="right")
    plt.axvline(x=1,linewidth=3, color='k', linestyle="--")
    plt.pause(5)
    plt.show()
