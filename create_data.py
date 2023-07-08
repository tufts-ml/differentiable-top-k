import argparse
import os

import numpy as np

from data_utils import sum_two_most_yellow



def main(num_samples=None, num_vals=None, out_dir=None, seed=None):

    # each val has an RGB component
    num_rgb_dims = num_vals*3

    num_dims = num_vals + num_rgb_dims

    # Generate synthetic data
    data = np.zeros((num_samples, num_dims))
    test_data = np.zeros((num_samples, num_dims))

    rng = np.random.default_rng(seed=seed)

    for i in range(num_dims):
        if i < num_vals:
            # Generate random values for numerical dimensions
            data[:, i] = rng.uniform(low=0, high=10.0, size=num_samples)
            test_data[:, i] = rng.uniform(low=0, high=10.0, size=num_samples)
        else:
            # Generate random values for RGB dimensions
            data[:, i] = rng.uniform(low=0.0, high=1.0, size=num_samples)
            test_data[:, i] = rng.uniform(low=0.0, high=1.0, size=num_samples)

    y_true = sum_two_most_yellow(data)
    y_test = sum_two_most_yellow(test_data)

    data_file = os.path.join(out_dir, 'x_data.csv')
    test_data_file = os.path.join(out_dir, 'x_test.csv')
    y_file = os.path.join(out_dir, 'y_data.csv')
    y_test_file = os.path.join(out_dir, 'y_test.csv')

    np.savetxt(data_file, data, delimiter=',')
    np.savetxt(y_file, y_true, delimiter=',')
    np.savetxt(test_data_file, test_data, delimiter=',')
    np.savetxt(y_test_file, y_test, delimiter=',')
    
    return data, y_true, test_data, y_test


if __name__ == '__main__':
    # create the parser
    parser = argparse.ArgumentParser(description='Create synthetic data')

    # add the arguments
    parser.add_argument('--num_samples', type=int, default=5000, help='number of samples')
    parser.add_argument('--num_vals', type=int, default=9, help='number of values')
    parser.add_argument('--out_dir', type=str, required=True, help='output directory')
    parser.add_argument('--seed', type=int, default=360, help='random seed')

    # parse the arguments
    args = parser.parse_args()

    main(**vars(args))
