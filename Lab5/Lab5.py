import scipy.stats as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

sizes = [20, 60, 100]
ro = [0, 0.5, 0.9]


def E(z):
    return np.mean(z)


def E_2(z):
    return np.mean(np.power(z, 2))


def D(z):
    return np.var(z)


def quadr(x, y):
    medx = np.median(x)
    medy = np.median(y)
    sum_e = 0
    for i in range(0, len(x)):
        sum_e = sum_e + np.sign(x[i] - medx) * np.sign(y[i] - medy)
    return sum_e / len(x)


def pearsonr(x, y):
    return st.pearsonr(x, y)[0]


def spearmanr(x, y):
    return st.spearmanr(x, y)[0]


cor_coef = {
    'pearson': pearsonr,
    'spearman': spearmanr,
    'quadr': quadr
}


def normal_dist(ro_e, N):
    x_mean = 0
    y_mean = 0
    std_x = 1
    std_y = 1
    cov = [[std_x**2, std_x*std_y*ro_e], [std_x*std_y*ro_e, std_y**2]]
    return st.multivariate_normal.rvs(mean=[x_mean, y_mean], cov=cov, size=N)


def mix_norm_dist(N):
    x_mean1 = 0
    y_mean1 = 0
    std_x1 = 1
    std_y1 = 1
    x_mean2 = 0
    y_mean2 = 0
    std_x2 = 10
    std_y2 = 10
    ro1 = 0.9
    ro2 = -0.9
    cov1 = [[std_x1**2, std_x1*std_y1*ro1], [std_x1*std_y1*ro1, std_y1**2]]
    cov2 = [[std_x2**2, std_x2*std_y2*ro2], [std_x2*std_y2*ro2, std_y2**2]]
    return 0.9 * st.multivariate_normal.rvs(mean=[x_mean1, y_mean1], cov=cov1, size=N) + 0.1 * st.multivariate_normal.rvs(mean=[x_mean2, y_mean2], cov=cov2, size=N)


def eigsorted(cov):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:, order]


file = open('out.txt', 'w')
for size in sizes:
    for coef in cor_coef.keys():
        file.write(coef + str(size) + '------------\n')
        for r in ro:
            vares = [normal_dist(r, size) for i in range(0, 1000)]
            vars_x = [[r[0] for r in var] for var in vares]
            vars_y = [[r[1] for r in var] for var in vares]
            cor_arr = [cor_coef[coef](vars_x[i], vars_y[i]) for i in range(0, len(vars_x))]
            file.write(str(r) + ': E=' + str(E(cor_arr)) + ' E_2=' + str(E_2(cor_arr)) + ' D=' + str(D(cor_arr)) + '\n')
        vares = [mix_norm_dist(size) for i in range(0, 1000)]
        vars_x = [[r[0] for r in var] for var in vares]
        vars_y = [[r[1] for r in var] for var in vares]
        cor_arr = [cor_coef[coef](vars_x[i], vars_y[i]) for i in range(0, len(vars_x))]
        file.write('for mixin: E=' + str(E(cor_arr)) + ' E_2=' + str(E_2(cor_arr)) + ' D=' + str(D(cor_arr)) + '\n')
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 12))
    ax = axes.flatten()
    for i in range(0, 4):
        if i != 3:
            points = normal_dist(ro[i], size)
            ax[i].set_title('n = ' + str(size) + ', r=' + str(ro[i]))
        else:
            points = mix_norm_dist(size)
            ax[i].set_title('n = ' + str(size) + ', mix')
        nstd = 2
        r_x = [point[0] for point in points]
        r_y = [point[1] for point in points]
        ax[i].plot(r_x, r_y, 'bo', ms=4)
        cov = np.cov(r_x, r_y)
        vals, vecs = eigsorted(cov)
        theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        w, h = 2 * nstd * np.sqrt(vals)
        ell = Ellipse(xy=(np.mean(r_x), np.mean(r_y)),
                      width=w, height=h,
                      angle=theta, color='black')
        ell.set_facecolor('none')
        ax[i].add_artist(ell)
        plt.tight_layout()
        fig.savefig('ellipse n=' + str(size))