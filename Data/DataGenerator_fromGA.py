import scipy.io
import numpy as np

N = 1024

name = 'N'+str(N)+'_frozenSet_GaussianApprox_v3.mat'
m = scipy.io.loadmat(name)["err_prob"] 
mat = m.T[0]
indexes = np.argsort(mat) #from the most reliable to the least

# Parameters
size = 19 * (10**3)
r = 256  # permutation range
filename = f"data_from_GA.txt"

# Generate unique sequences
unique_sequences = set()

# First sample (GA)
binary = np.ones(N, dtype=int)
binary[indexes] = 0
unique_sequences.add(' '.join(map(str, binary)))

# Generate remaining sequences
while len(unique_sequences) < size:
    temp = indexes.copy()
    np.random.shuffle(temp[N//2-r:N//2+r])
    binary = np.ones(N, dtype=int)
    binary[temp[:N//2]] = 0
    sequence = ' '.join(map(str, binary))
    unique_sequences.add(sequence)

# Write sequences to file
with open(filename, 'w') as f:
    for sequence in unique_sequences:
        f.write(f"{sequence}\n")

print(f"Number of unique sequences generated and written to {filename}: {len(unique_sequences)}")