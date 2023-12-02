import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Set the parameters
T_FINAL = 150
N_MAX = 10
N_INITIAL = [1, 5, 9]

def coexistence_game(n, t, N):
    """
    Zero-fluctuation approximation of "coexistence game" random walk.

    n = 0, 1, ..., N
    """
    dndt = N**-3 * (N**2 * n - 3 * N * n**2 + 2 * n**3)
    return dndt

# Set the time points at which to solve the ODEs
t_numerical = np.linspace(0, T_FINAL, 1000)


# Solve the ODEs numerically for each initial condition:
for i, n_initial in enumerate(N_INITIAL):
    solution = odeint(coexistence_game, n_initial, t_numerical, args=(N_MAX,))
    plt.plot(t_numerical, *solution.T, label=n_initial)

plt.title("Coexistence Game Steady State (Ignoring Fluctuations)")
plt.legend(title="Initial $n$")
plt.show()


