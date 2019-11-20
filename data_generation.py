from sklearn.datasets import make_classification
import scipy.io
import numpy as np
import sys
import os

def mmwrite_piece_data(piece_array_list, save_dir, filename_marker):
    array_num = len(piece_array_list)
    for array_idx in range(array_num):
        filename = "%s%d.dat"%(filename_marker, array_idx + 1)
        filename_dir = os.path.join(save_dir, filename)
        scipy.io.mmwrite(target = filename_dir, a = piece_array_list[array_idx])
        if os.path.isfile(filename_dir):
            os.remove(filename_dir)
        os.rename("%s.mtx"%filename_dir, filename_dir) 
    return 0

def main(cores_num, sample_num, feature_num):
    save_dir = os.path.join(".", "data")
    
    logit_data = make_classification(n_samples = sample_num, n_features = feature_num)
    
    A = logit_data[0]
    B = logit_data[1].reshape(-1, 1)
    
    A_split = np.array_split(A, cores_num, axis = 0)
    B_split = np.array_split(B, cores_num, axis = 0)
    
    A_marker = "A"
    B_marker = "b"
    mmwrite_piece_data(A_split, save_dir, A_marker)
    mmwrite_piece_data(B_split, save_dir, B_marker)
    return 0

if __name__ == '__main__':
	cores_num = int(sys.argv[1])
	sample_num = int(sys.argv[2])
	feature_num = int(sys.argv[3])
	main(cores_num, sample_num, feature_num)
