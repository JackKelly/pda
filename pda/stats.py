from __future__ import print_function, division
from scipy.stats import linregress

def correlate(x, y, ax, xlabel='', ylabel=''):
    x = x.dropna()
    y = y.dropna()
    x_aligned, y_aligned = x.align(y, join='inner')
    print('aligned data starts {}, ends {}, length = {}'
          .format(x_aligned.index[0], x_aligned.index[-1], 
                  x_aligned.size))
    ax.plot(x_aligned, y_aligned, 'o', alpha=0.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    # calculate linear regression
    slope, intercept, r_value, p_value, std_err = linregress(x_aligned.values,
                                                             y_aligned.values)

    # plot linear regression line
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.plot(xlim, [intercept, intercept+(xlim[1]*slope)], 'k-')
    ax.annotate('$R^2 = {:.3f}$'.format(r_value**2), ((xlim[1]-xlim[0])*0.8,
                                                  (ylim[1]-ylim[0])*0.8))
    ax.set_ylim((0, ylim[1]))
