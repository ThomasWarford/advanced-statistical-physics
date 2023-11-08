import numpy as np
import matplotlib.pyplot as plt

N = 100_000
decay_rate = 2

X = np.random.rand(N)
Y = - (1 / decay_rate) * np.log(X)

time = np.linspace(0, 15, 1000)
exponential_pdf = decay_rate * np.exp(-decay_rate*time)

plt.plot(time, exponential_pdf)
plt.hist(Y, density=True)

plt.show()