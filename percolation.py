import numpy as np
from collections import defaultdict
from numba import jit
import matplotlib.pyplot as plt


results1 = defaultdict(lambda: defaultdict(int))
results2 = dict()

@jit(nopython=True)
def is_cluster(L: int, p: float) -> dict:
    M = np.array([[0  if np.random.random() < p else 1 for i in range(L)] for j in range(L)])
    M[0] = [2 if i == 0 else 1 for i in M[0]]

    for i in range(1, L):
        for j in range(L):
            if M[i, j] == 0 and (2 in [M[i - 1, j], M[i, j - 1]]):
                M[i, j] = 2

    return 2 in M[-1, :]


for L in [8, 16, 32, 64, 128]:
    for i in np.linspace(0, 1, 101):
        for _ in range(1000):
            if is_cluster(L, i):
                results1[L][i] += 1
            else:
                results1[L][i] += 0
    results2[L] = max(filter(lambda p: p[1] == 0, results1[L].items()))[0]



fig, ax = plt.subplots()
for L in [8, 16, 32, 64, 128]:
    ax.plot(results1[L].keys(), results1[L].values(), label=f'L = {L}')
ax.legend()
ax.set(xlabel='Вероятность', ylabel='N', title=f'Число обнаружения стягивающих кластеров среди 1000 испытаний для решеток разных размерностей')
ax.grid()
plt.show()


fig, ax = plt.subplots()
plt.xticks(list(results2.keys()))
plt.yticks(list(results2.values()))
ax.plot(results2.keys(), results2.values(), marker='o')
ax.set(xlabel='L', ylabel='Порог протекания', title='Зависимость порога протекания от размера решетки L')
plt.show()



