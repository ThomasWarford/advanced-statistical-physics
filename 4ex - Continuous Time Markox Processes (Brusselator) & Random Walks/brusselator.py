import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Set the parameters
T_FINAL = 20
B = 1.8 # rate of X->Y per X
SYSTEM_SIZE = 1000 # aka Omega
INITIAL_CONDITIONS = [0, 0] # [n_x, n_y]

## Numerical Integration:
# Define the Brusselator system of ODEs
def brusselator(r, t, B):
    x, y = r
    dxdt = 1 - (B + 1) * x + x**2 * y
    dydt = B * x - x**2 * y
    return [dxdt, dydt]

# Set the time points at which to solve the ODEs
t_numerical = np.linspace(0, T_FINAL, 1000)

# Solve the ODEs numerically
solution = odeint(brusselator, INITIAL_CONDITIONS, t_numerical, args=(B,))

# Extract the results for x and y
x_numerical, y_numerical = solution.T

## Gillespie Algorithm Simulation:

def draw_jump_time(jump_rate_total):
    '''
    This function draws the time of the first jump using inverse transform sampling.

    Time to first jump is modeled with an exponential distribution.
    '''
    r = np.random.uniform()
    return - (1 / jump_rate_total) * np.log(r)

t_gillespie = []
x_gillespie = []
y_gillespie = []
time = 0
n_x = 0
n_y = 0

while time < T_FINAL:
    t_gillespie.append(time)
    x_gillespie.append(n_x/SYSTEM_SIZE)
    y_gillespie.append(n_y/SYSTEM_SIZE)

    total_rate = (
        SYSTEM_SIZE # from Xs "appearing" from vacuum
        + n_x # rate of Xs decaying
        + B * n_x # rate of X -> Y
        + SYSTEM_SIZE**-2 * n_x**2 * n_y # rate of 2X+Y->3X
    )

    time += draw_jump_time(total_rate)

    uniform_draw = np.random.uniform()

    if uniform_draw < SYSTEM_SIZE/total_rate:
        n_x += 1
    elif uniform_draw < (SYSTEM_SIZE+n_x)/total_rate:
        n_x -= 1
    elif uniform_draw < (SYSTEM_SIZE+n_x+B*n_x)/total_rate:
        n_x -= 1
        n_y += 1
    else:
        n_x += 1
        n_y -= 1


# Plot the results
plt.plot(t_numerical, x_numerical, label='X')
plt.plot(t_gillespie, x_gillespie)
plt.plot(t_numerical, y_numerical, label='Y')
plt.plot(t_gillespie, y_gillespie)
plt.xlabel('Time')
plt.ylabel(r'Concentration ($\frac{n}{\Omega}$)')
plt.title(rf'Brusselator System, $b={B}$, $\Omega={SYSTEM_SIZE}$')
plt.savefig(rf"b{B}_Omega{SYSTEM_SIZE}.png")
plt.legend()
plt.show()