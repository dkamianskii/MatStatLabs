import scipy.stats as st
import numpy as np
import seaborn as sns
import pandas as pd
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
    sum = 0
    for i in range(r, n - r):
        sum += x[i]
    return sum/(n - 2 * r)


pos_characteristics = {
    'average': np.mean,
    'med': np.median,
    'Zr': Zr,
    'Zq': Zq,
    'Ztr': Ztr
}

sizes = [20, 100]

def IQR(x):
    return np.quantile(x, 3 / 4) - np.quantile(x, 1 / 4)


def ejections(X):
    iqr = IQR(X)
    X1 = np.quantile(X, 1 / 4) - 1.5*iqr
    X2 = np.quantile(X, 3 / 4) + 1.5*iqr
    count = 0
    for x in X:
        if (x < X1) or (x > X2):
            count += 1
    return count


theor_pecnt_of_ejections = []
for dist in distributions.keys():
    iqr = distributions[dist].ppf(0.75) - distributions[dist].ppf(0.25)
    X1 = distributions[dist].ppf(0.25) - 1.5 * iqr
    X2 = distributions[dist].ppf(0.75) + 1.5 * iqr
    if(dist == 'poisson'):
        theor_pecnt_of_ejections.append(distributions[dist].cdf(X1) - distributions[dist].pmf(X1) + 1 - distributions[dist].cdf(X2))
    else:
        theor_pecnt_of_ejections.append(distributions[dist].cdf(X1) + 1 - distributions[dist].cdf(X2))


data = [dist.rvs(20) for dist in distributions.values()]
df = pd.DataFrame(list(zip(*data)), columns=[key for key in distributions.keys()])
fig, ax = plt.subplots(figsize=(10, 8))
sns_plot = sns.boxplot(data=df, orient='v', ax=ax)
figu = sns_plot.get_figure()
figu.savefig("boxplot_N=" + str(20) + ".png")

for dist in distributions.keys():
    fig, ax = plt.subplots(figsize=(10, 8))
    sns_plot = sns.boxplot(data=distributions[dist].rvs(100), orient='h', ax=ax)
    figu = sns_plot.get_figure()
    figu.savefig("boxplot_N=100_" + dist + ".png")

file = open('out.txt', 'w')
tmp = 0
for dist in distributions.keys():
    file.write(dist + '\n')
    file.write('theoretical percent of ejection:  ' + str(theor_pecnt_of_ejections[tmp]) + '\n')
    tmp += 1
    file.write('percent of ejection experimental:  ')
    for size in sizes:
        varis = [ejections(distributions[dist].rvs(size))/size for i in range(0, 1000)]
        percent_of_ejections = np.mean(varis)
        variance = np.var(varis)
        file.write('N=' + str(size) + ',  D=' + str(variance) + ' :     ')
        file.write(str(percent_of_ejections) + '     ')
    file.write('\n\n')




