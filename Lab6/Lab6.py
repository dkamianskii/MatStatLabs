import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import dmatrices


def y_acc(x_e, err):
    return 2 + 2*x_e + err


x_pr = np.arange(-1.8, 2.2, 0.2)
e = st.norm(loc=0, scale=1).rvs(20)
y_pr = np.array([y_acc(x_pr[i], e[i]) for i in range(x_pr.size)])
data = {'x': x_pr,
        'y': y_pr}

y, x = dmatrices("y ~ x", data, return_type='dataframe')

mod_ls = sm.OLS(y, x)
mod_lad = smf.quantreg('y ~ x', data)
ls_res = mod_ls.fit()
lad_res = mod_lad.fit(q=.5)

file = open('out.txt', 'w')
file.write('ls coefs :  ' + str(ls_res.params) + '\n')
file.write('lad coefs :  ' + str(lad_res.params) + '\n')

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 8))

ax[0].plot(x_pr, y_pr, 'o', label="Data")
ax[0].plot(x_pr, ls_res.predict(), 'r-', label="LSM")
ax[0].plot(x_pr, lad_res.predict(), 'b-', label="LADM")
ax[0].plot(x_pr, (2 + 2*x_pr), 'g-', label="y = 2 + 2x")
ax[0].legend(loc="best")
ax[0].set_title('original data sample')

file.write('results with distortion in y\n')
y_pr[0] += 10
y_pr[19] += -10

data = {'x': x_pr,
        'y': y_pr}

y, x = dmatrices("y ~ x", data, return_type='dataframe')

mod_ls = sm.OLS(y, x)
mod_lad = smf.quantreg('y ~ x', data)
ls_res = mod_ls.fit()
lad_res = mod_lad.fit(q=.5)

file.write('ls coefs :  ' + str(ls_res.params) + '\n')
file.write('lad coefs :  ' + str(lad_res.params) + '\n')

ax[1].plot(x_pr, y_pr, 'o', label="Data")
ax[1].plot(x_pr, ls_res.predict(), 'r-', label="LSM")
ax[1].plot(x_pr, lad_res.predict(), 'b-', label="LADM")
ax[1].plot(x_pr, (2 + 2*x_pr), 'g-', label="y = 2 + 2x")
ax[1].set_title('data sample with distortion')
ax[1].legend(loc="best")

fig.savefig('graph')
