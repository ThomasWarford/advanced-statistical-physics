import numpy as np
import matplotlib.pyplot as plt

NUM_TRIALS = 100_000
N = 100 # Maximum integer labelling state
n = 10 # Initial state

jump_rate_plus = 0.05 # rate of jumps to i+1
jump_rate_minus = 0.05 # rate of jumps to i-1

jump_rate_total = jump_rate_plus + jump_rate_minus

def draw_jump_time(jump_rate_total, size=1):
    '''
    This function draws the time of the first jump using inverse transform sampling.

    Time to first jump is modeled with an exponential distribution.
    '''
    r = np.random.uniform(size=size)
    return - (1 / jump_rate_total) * np.log(r)


positions = np.ones(NUM_TRIALS, dtype=np.int32) * n
fixation_times = np.zeros(NUM_TRIALS)

active_trials = (0<positions) & (positions<N)

while active_trials.any():
    jump_times = draw_jump_time(jump_rate_total, NUM_TRIALS)

    # update fixation times unless at fixation already
    fixation_times[active_trials] += jump_times[active_trials]

    # change position based on value of random number
    r = np.random.uniform(size=NUM_TRIALS)

    jump_probability_plus = jump_rate_plus/jump_rate_total
    positions[(r<=jump_probability_plus) & active_trials] += 1
    positions[(r>jump_probability_plus) & active_trials] -= 1

    active_trials = (0<positions) & (positions<N)


fig, ax = plt.subplots(3, 1)

fig.suptitle("Unconditional and Conditional Fixation Time Distributions")
n_bins = 32

ax[0].hist(fixation_times, density=True, bins=n_bins)
ax[0].set_xlabel("Unconditional Fixation Time")

ax[1].hist(fixation_times[positions==0], density=True, bins=n_bins)
ax[1].set_xlabel("n=0 Fixation Time")

ax[2].hist(fixation_times[positions==N], density=True, bins=n_bins)
ax[2].set_xlabel("n=N Fixation Time")

fig.tight_layout()
fig.savefig("out.png")
plt.show()

print("Mean Unconditional Fixation Time:", np.mean(fixation_times))
print("Mean n=0 Fixation Time:", np.mean(fixation_times[positions==0]))
print("Mean n=N Fixation Time:", np.mean(fixation_times[positions==N]))
print("Probability of Absorption at n=N:", (positions==N).sum()/len(positions))
