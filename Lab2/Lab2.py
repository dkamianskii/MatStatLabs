import scipy.stats as st
import numpy as np

distributions = {
    'normal': st.norm(loc=0, scale=1),
    'laplace': st.laplace(loc=0, scale=1/np.sqrt(2)),
    'cauchy': st.cauchy(),
    'uniform': st.uniform(loc=-np.sqrt(3), scale=2*np.sqrt(3)),
    'poisson': st.poisson(5)
}

sizes = [10, 100, 1000]

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

def E(z):
    return np.mean(z)


def D(z):
    return np.var(z)


file = open('out.txt', 'w')

for dist in distributions.keys():
    file.write('-------------------------------------\n')
    file.write(dist + '\n')
    for size in sizes:
        E_arr = []
        D_arr = []
        for char in pos_characteristics.keys():
            arr = [pos_characteristics[char](var) for var in [distributions[dist].rvs(size=size) for i in range(0, 1000)]]
            E_arr.append(E(arr))
            D_arr.append(D(arr))
            file.write(char + '  ')

        file.write('\n')
        [file.write(str(e) + '  ') for e in np.around(E_arr, 2)]
        file.write('\n')
        [file.write(str(d) + '  ') for d in np.around(D_arr, 6)]
        file.write('\n')
