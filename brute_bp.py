import pandas as pd
import numpy as np
import time
from numpy import binary_repr
import multiprocessing as mp
import matplotlib.pyplot as plt


def brute_bp(capacity, wts, vals, item_count, start, end):
    best_value = 0

    for i in range(start, end):
        binary = binary_repr(i, width=item_count)

        my_list_combinations = (tuple(binary))

        sum_weight = 0
        sum_value = 0

        for wt in wts:
            index = np.where(wts == wt)[0][0]
            sum_weight += int(my_list_combinations[index]) * wt

        if sum_weight <= capacity:
            for v in vals:
                index_v = np.where(vals == v)[0][0]
                sum_value += int(my_list_combinations[index_v]) * v
            if sum_value > best_value:
                best_value = sum_value
    return best_value


def task(args):
    pid = args[0]
    num_threads = args[1]
    values = args[2]
    capacity = args[3]
    wts = values[:, 0]
    vals = values[:, 1]
    item_count = values.shape[0]
    S = pow(2, item_count)
    s = S // num_threads
    start = s * pid
    count = s if (pid != (num_threads - 1)) else s + S % num_threads
    max_price = brute_bp(capacity, wts, vals, item_count, start, start + count)
    return max_price


if __name__ == '__main__':
#    tests = ["5", "8", "18", "20"]
    tests = ["15"]

    for t in tests:
        threads = [i for i in range(1, 9)]
        times = []

        for thr in threads:
            start_time = time.time()
            goods = pd.read_csv(r"tests\test_" + t + ".csv", header=None, delimiter=";").values
            W = sum(goods[:, 0]) / 2
            with mp.Pool(thr) as p:
                res = max(p.map(task, [(i, thr, goods, W) for i in range(thr)]))
            times.append(time.time() - start_time)
        accs = list(map(lambda x: x / times[0], times))
        plt.plot(threads, accs, label=t)
    plt.legend()
    plt.show()

