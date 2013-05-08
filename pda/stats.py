from __future__ import print_function, division

def align(x, y):
    xdescription = x.__dict__.get('description')
    ydescription = y.__dict__.get('description')
    x = x.dropna()
    y = y.dropna()
    x_aligned, y_aligned = x.align(y, join='inner')
    print('aligned data starts {}, ends {}, length = {}'
          .format(x_aligned.index[0], x_aligned.index[-1], 
                  x_aligned.size))
    x_aligned.description = xdescription
    y_aligned.description = ydescription
    return x_aligned, y_aligned

def plot_regression_line(ax, x_aligned, y_aligned, slope, intercept, r_value):
    ax.plot(x_aligned, y_aligned, 'o', alpha=0.2)
    # plot linear regression line
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.plot(xlim, [intercept+(xlim[0]*slope), intercept+(xlim[1]*slope)], 'k-')
    ax.annotate('$R^2 = {:.3f}$\n'
                '$n = {:d}$'
                .format(r_value**2, x_aligned.size),
                ((xlim[1]-xlim[0])*0.8 + xlim[0],
                 (ylim[1]-ylim[0])*0.8))
    ax.set_ylim((0, ylim[1]))
    return ax
