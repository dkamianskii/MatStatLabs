import scipy.stats as st
import numpy as np

gamma = 0.95
norm = st.norm(loc=0, scale=1)

sizes = [20, 100]


def conf_int_m(x, alpha):
    x_mean = x.mean()
    s = x.std()
    n = x.size
    t_a = st.t(df=(n - 1)).ppf(1 - alpha/2)
    temp = s*t_a/np.sqrt(n - 1)
    return x_mean - temp, x_mean + temp


def conf_int_sigma(x, alpha):
    s = x.std()
    n = x.size
    hi_1_a = st.chi2(df=(n - 1)).ppf(1 - alpha/2)
    hi_a = st.chi2(df=(n - 1)).ppf(alpha / 2)
    return s*np.sqrt(n)/np.sqrt(hi_1_a), s*np.sqrt(n)/np.sqrt(hi_a)


def conf_ass_m(x, alpha):
    u = norm.ppf(1 - alpha/2)
    x_mean = x.mean()
    s = x.std()
    n = x.size
    temp = s*u/np.sqrt(n)
    return x_mean - temp, x_mean + temp

def conf_ass_sigma(x, alpha):
    s = x.std()
    n = x.size
    e = st.moment(x, 4)/(s**4) - 3
    U = norm.ppf(1 - alpha/2)*np.sqrt((e + 2)/n)
    return s/np.sqrt(1 + U), s/np.sqrt(1 - U)



file = open('out.txt', 'w')
for size in sizes:
    x_v = norm.rvs(size)
    file.write('estimations for sample size = ' + str(size) + '\n')
    file.write('for normal dist\n')
    file.write('m = ' + str(conf_int_m(x_v, 1 - gamma)) + '  sigma = ' + str(conf_int_sigma(x_v, 1 - gamma)) + '\n')
    file.write('for random dist asymptote method\n')
    file.write('m = ' + str(conf_ass_m(x_v, 1 - gamma)) + '  sigma = ' + str(conf_ass_sigma(x_v, 1 - gamma)) + '\n')