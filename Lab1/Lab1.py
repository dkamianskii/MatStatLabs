import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st


size = [10, 50, 100, 1000]

distributions = {
    'normal': st.norm(loc=0, scale=1),
    'laplace': st.laplace(loc=0, scale=1/np.sqrt(2)),
    'cauchy': st.cauchy(),
    'uniform': st.uniform(loc=-np.sqrt(3), scale=np.sqrt(3)),
}

#GRAPHICS

for dist in distributions.keys():
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
    ax = axes.flatten()
    for i in range(0, 4):
        x = np.linspace(distributions[dist].ppf(0.01), distributions[dist].ppf(0.99), 100)
        ax[i].plot(x, distributions[dist].pdf(x), 'r-', lw=4)
        r = distributions[dist].rvs(size=size[i])
        ax[i].hist(r, density=True, bins=20, histtype='stepfilled', alpha=0.2)
        ax[i].set_title('sample size =' + str(size[i]))
        ax[i].grid()
        fig.savefig(dist)



mu = 5
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
ax = axes.flatten()
for i in range(0, 4):
    x = np.arange(st.poisson.ppf(0.01, mu), st.poisson.ppf(0.99, mu))
    ax[i].plot(x, st.poisson.pmf(x, mu), 'ro', ms=6)
    ax[i].vlines(x, 0, st.poisson.pmf(x, mu), colors='r', lw=4, alpha=0.2)
    r = st.poisson.rvs(mu, size=size[i])
    ax[i].hist(r, density=True, bins=20, histtype='stepfilled', alpha=0.2)
    ax[i].set_title('sample size =' + str(size[i]))
    ax[i].grid()
    fig.savefig('poisson')
