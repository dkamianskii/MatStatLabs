import scipy.stats as st
import numpy as np

# N_t = [100, 1000]
#
#
# file = open('out.txt', 'w')
#
norm = st.norm(loc=0, scale=1)
# x = norm.rvs(N)

# m = x.mean()
# sigma = x.std()
# file.write('m = ' + str(m) + 'sigma =  ' + str(sigma) + '\n')
# for N_test in N_t:
#     file.write('N_test = ' + str(N_test) + '\n')
#     k = int(1.72 * np.cbrt(N_test))
#     rs = norm.rvs(N_test)
#
#     if k % 2 == 1:
#         k -= 1
#
#     left = -1
#     right = 1
#     step = (right - left) / (k - 2)
#
#     delta = [left + i * step for i in range(0, k - 1)]
#
#     n = np.zeros(k)
#     for r in rs:
#         last = True
#         for i in range(0, len(delta)):
#             if r < delta[i]:
#                 n[i] += 1
#                 last = False
#                 break
#         if last:
#             n[k - 1] += 1
#
#     p = []
#     model = st.norm(loc=m, scale=sigma)
#
#     for i in range(0, k - 1):
#         if i == 0:
#             p.append(model.cdf(delta[0]))
#         else:
#             p.append(model.cdf(delta[i]) - model.cdf(delta[i - 1]))
#
#     p.append(1 - model.cdf(delta[k - 2]))
#     p = np.array(p)
#     # p = np.around(p, 5)
#     print(p.sum())
#
#     hi_square = 0
#     for i in range(0, k):
#         file.write('i = ' + str(i))
#         if i == k - 1:
#             file.write('  delta = inf')
#         else:
#             file.write('  delta = ' + str(delta[i]))
#         file.write('  n = ' + str(n[i]) + '  p_i = ' + str(p[i]))
#         temp_v = ((n[i] - N_test * p[i]) ** 2) / (N_test * p[i])
#         file.write('  temp_v = ' + str(temp_v) + '\n')
#         hi_square += temp_v
#
#     file.write('\n')
#     file.write('hi_square = ' + str(hi_square))
#     file.write('\n')


N_laplace = 25
laplace = st.laplace(loc=0, scale=1/np.sqrt(2))
x_laplace = laplace.rvs(N_laplace)
file = open('laplace2.txt', 'w')

m = x_laplace.mean()
sigma = x_laplace.std()
file.write('m = ' + str(m) + 'sigma =  ' + str(sigma) + '\n')
k = int(1.72 * np.cbrt(N_laplace))


if k % 2 == 1:
    k -= 1

left = -0.2
right = 0.2
step = (right - left) / (k - 2)

delta = [left + i * step for i in range(0, k - 1)]

n = np.zeros(k)
for r in x_laplace:
    last = True
    for i in range(0, len(delta)):
        if r < delta[i]:
            n[i] += 1
            last = False
            break
    if last:
        n[k - 1] += 1

p = []
model = st.norm(loc=m, scale=sigma)

for i in range(0, k - 1):
    if i == 0:
        p.append(model.cdf(delta[0]))
    else:
        p.append(model.cdf(delta[i]) - model.cdf(delta[i - 1]))

p.append(1 - model.cdf(delta[k - 2]))
p = np.array(p)
print(p.sum())

hi_square = 0
for i in range(0, k):
    file.write('i = ' + str(i))
    if i == k - 1:
        file.write('  delta = inf')
    else:
        file.write('  delta = ' + str(delta[i]))
    file.write('  n = ' + str(n[i]) + '  p_i = ' + str(p[i]))
    temp_v = ((n[i] - N_laplace * p[i]) ** 2) / (N_laplace * p[i])
    file.write('  temp_v = ' + str(temp_v) + '\n')
    hi_square += temp_v

file.write('\n')
file.write('hi_square = ' + str(hi_square))
file.write('\n')
