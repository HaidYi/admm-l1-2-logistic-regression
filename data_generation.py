from sklearn.datasets import make_classification
import scipy.io
import numpy as np
import sys
import os
import shutil

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

def divide_logit_data(observations, labels, cores_num):
    obs_split = np.array_split(observations, cores_num, axis = 0)
    lab_split = np.array_split(labels, cores_num, axis = 0)
    return obs_split, lab_split

def str2list(num_str):
    str_list = num_str.split(",")
    num_list = list(map(int, str_list))
    return num_list

def save_dir_generator(cores_num, sample_num, feature_num, save_root, save_dirname):
    save_dir = os.path.join(save_root, "%s_%d_%d_%d"%(save_dirname, sample_num, feature_num, cores_num))
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.mkdir(save_dir)
    return save_dir

def seq_save_dir_generator(sample_num, feature_num, save_root, seq_save_dirname):
    seq_save_dir = os.path.join(save_root, "%s_%d_%d"%(seq_save_dirname, sample_num, feature_num))
    if os.path.exists(seq_save_dir):
        shutil.rmtree(seq_save_dir)
    os.mkdir(seq_save_dir)
    return seq_save_dir

def main(cores_num_str, sample_num, feature_num, save_root):
    A_marker = "A"
    B_marker = "b"
    save_dirname = "data"
    cores_num_list = str2list(cores_num_str)
    
    logit_data = make_classification(n_samples = sample_num, n_features = feature_num)
    A = logit_data[0]
    B = logit_data[1].reshape(-1, 1)
    
    for cores_num in cores_num_list:
        save_dir = save_dir_generator(cores_num, sample_num, feature_num, save_root, save_dirname)
        A_split = np.array_split(A, cores_num, axis = 0)
        B_split = np.array_split(B, cores_num, axis = 0)
    
        mmwrite_piece_data(A_split, save_dir, A_marker)
        mmwrite_piece_data(B_split, save_dir, B_marker)
    
    seq_input_filename = "input"
    seq_output_filename = "output"
    seq_save_dirname = "data_seq"
    seq_save_dir = seq_save_dir_generator(sample_num, feature_num, save_root, seq_save_dirname)
    seq_input_dir_filename = os.path.join(seq_save_dir, seq_input_filename)
    seq_output_dir_filename = os.path.join(seq_save_dir, seq_output_filename)
    scipy.io.mmwrite(target = seq_input_dir_filename, a = A)
    scipy.io.mmwrite(target = seq_output_dir_filename, a = B)
    return 0

if __name__ == '__main__':
    cores_num_str = sys.argv[1]
    sample_num = int(sys.argv[2])
    feature_num = int(sys.argv[3])
    save_root = sys.argv[4]
    main(cores_num_str, sample_num, feature_num, save_root)
