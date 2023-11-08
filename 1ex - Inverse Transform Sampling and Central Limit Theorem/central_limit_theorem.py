import numpy as np
import matplotlib.pyplot as plt

# number making up each batch
ns_per_batch = [100, 10, 2, 1]
number_of_batches_per_n = 100_000

def get_averages_from_batch(n_per_batch, number_of_batches_per_n):

    rand_numbers = np.random.rand(number_of_batches_per_n, n_per_batch)
    return rand_numbers.mean(axis=1)


sigma = np.sqrt(1/12)
print("sigma (for uniform draw):", sigma)

for n in ns_per_batch:
    means = get_averages_from_batch(n, number_of_batches_per_n)
    plt.hist(means, density=True)
    print("n:", n, "| sigma/sqrt(n):", sigma/np.sqrt(n), "| calculated std:", np.std(means))

plt.show()