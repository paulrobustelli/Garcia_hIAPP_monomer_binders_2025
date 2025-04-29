import numpy as np 
from scipy.sparse import csr_matrix, save_npz
import sys
from os.path import abspath 

dense_path = sys.argv[1] 
sparse_path = sys.argv[2]

circuit_matrix = np.load(abspath(dense_path))
sparse_circuit_matrix = csr_matrix(circuit_matrix)
save_npz(abspath(sparse_path), sparse_circuit_matrix)


