import scipy.stats as st
import numpy as np
import matplotlib.pyplot as plt

distributions = {
    'normal': st.norm(loc=0, scale=1),
    'laplace': st.laplace(loc=0, scale=1/np.sqrt(2)),
    'cauchy': st.cauchy(),
    'uniform': st.uniform(loc=-np.sqrt(3), scale=2*np.sqrt(3)),
    'poisson': st.poisson(5)
}

def Zr(x):
    return (np.amin(x) + np.amax(x))/2


def Zq(x):
    return (np.quantile(x, 1/4) + np.quantile(x, 3/4))/2


def Ztr(x):
    n = x.size
    r = (int)(n / 4)
    sum1 = 0
    for i in range(r, n - r):
        sum1 += x[i]
    return sum1 / (n - 2 * r)


pos_characteristics = {
    'average': np.mean,
    'med': np.median,
    'Zr': Zr,
    'Zq': Zq,
    'Ztr': Ztr
}

sizes = [20, 60, 100]
k_h = [0.5, 1, 2]

def prob_dist_table(x_e):
    a = np.sort(x_e)
    result = []
    elms, counts = np.unique(a, return_counts=True)
    result.append(elms)
    dist_table = [counts[i] / x_e.size for i in range(0, elms.size)]
    sum_p = 0
    dist = []
    for p in dist_table:
        sum_p += p
        dist.append(sum_p)
    result.append(dist)
    return result


def prob_dist_func(x_e, table):
    for t in range(0, table[0].size):
        if t == 0 and x_e < table[0][t]:
            return 0
        if x_e < table[0][t]:
            return table[1][t - 1]
    return 1


def h(n, var):
    return 1.06*np.std(var)*np.power(n,-0.2)


def K(u):
    return np.exp(-(u**2)/2)/np.sqrt(2*np.pi)


def density_function(x_e, var, h_n_e):
    sum_d = 0
    for x_i in var:
        sum_d += K((x_e - x_i) / h_n_e)
    return sum_d / (var.size * h_n_e)


for size in sizes:
    for dist in distributions.keys():
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 8))
        r = distributions[dist].rvs(size=size)
        for i in range(0, 3):
            h_n = k_h[i]*h(r.size, r)
            if (dist == 'poisson'):
                x = [k for k in range(6, 15)]
                ax[i].plot(x, distributions[dist].pmf(x), 'bo', ms=4)
                ax[i].vlines(x, 0, distributions[dist].pmf(x), colors='b', lw=2, label='PDF', alpha=0.2)
                ax[i].plot(x, density_function(x, r, h_n), 'r-', label='Kernel density', lw=4)
            else:
                x = np.linspace(-4, 4, 100)
                ax[i].plot(x, distributions[dist].pdf(x), 'b-', label='PDF', lw=2)
                ax[i].plot(x, density_function(x, r, h_n), 'r-', label='Kernel density', lw=4)
            if k_h[i] != 1:
                ax[i].set_title(str(k_h[i]) + '*h_n')
            else:
                ax[i].set_title('h_n')
            ax[i].legend(loc='upper left')
            ax[i].grid()
        fig.suptitle('n=' + str(size) + 'with window width h =[0.5, 1, 2]', fontsize=12)
        fig.savefig(dist + '_pdf_' + str(size))

for dist in distributions.keys():
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 8))
    for i in range(0, 3):
        r = distributions[dist].rvs(size=sizes[i])
        table = prob_dist_table(r)
        if dist == 'poisson':
            x = [k for k in range(6, 15)]
            ax[i].plot(x, distributions[dist].cdf(x), 'b-', ms=4)
            ax[i].vlines(x, 0, distributions[dist].cdf(x), colors='b', lw=2, label='CDF', alpha=0.2)
            ax[i].plot(x, [prob_dist_func(x_i, table) for x_i in x], 'r-', label='Experimental distribution', lw=4)
        else:
            x = np.linspace(-4, 4, 100)
            ax[i].plot(x, distributions[dist].cdf(x), 'b-', label='CDF', lw=2)
            ax[i].plot(x, [prob_dist_func(x_i, table) for x_i in x], 'r-', label='Experimental distribution', lw=4)
        ax[i].set_title('n = ' + str(sizes[i]))
        ax[i].legend(loc='upper left')
        ax[i].grid()
    fig.savefig(dist + '_cdf')