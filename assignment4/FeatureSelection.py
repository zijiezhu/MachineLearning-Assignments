from scipy import sparse
import numpy as np
import csv
import math

try:
    import Queue as Q
except ImportError:
    import queue as Q

class FeatureValue(object):
    def __init__(self, info_gain, attr_Id):
        self.info_gain = info_gain
        self.attr_Id = attr_Id
    def __lt__(self, other):
        return other.info_gain < self.info_gain
    def __gt__(self, other):
        return other.info_gain > self.info_gain
    def __le__(self, other):
        return other.info_gain <= self.info_gain
    def __ge__(self, other):
        return other.info_gain >= self.info_gain

def parse_labels(file_name):
    lines = csv.reader(open(file_name, "r"))
    dataset = list(lines)
    binary_labels = []
    for i in range(len(dataset)):
        dataset[i] = [int(x) for x in dataset[i]]
        label = dataset[i][0]
        if label >= 12 and label <= 15:
            binary_labels.append(1)
        else:
            binary_labels.append(0)

    return binary_labels

def parse_samples(file_name):
    row = []
    col = []
    data = []
    lines = csv.reader(open(file_name, "r"))
    dataset = list(lines)

    for i in range(len(dataset)):
        datas = dataset[i][0].split(" ")
        cur_row = int(datas[0]) - 1
        cur_col = int(datas[1]) - 1
        cur_data = int(datas[2])

        row.append(cur_row)
        col.append(cur_col)
        data.append(cur_data)

    matrix = sparse.coo_matrix((data, (row, col)), shape=(11269, 61188)).toarray()
    return matrix

def cocat(labels, matrix):
    return np.c_[labels, matrix]

def get_range(labels):
    ranges = []
    start = 0
    end = 0
    for i in range(len(labels)):
        if i > 0 and labels[i] == 1 and labels[i - 1] == 0:
            start = i
            continue
        if labels[i] == 1 and labels[i + 1] == 0:
            end = i
    ranges.append(start)
    ranges.append(end)
    return ranges

def calc_entropy(N1, N):
    if N == 0 or N1 == 0:
        return 0
    N2 = N - N1
    temp1 = N1 / N
    temp2 = N2 / N
    if temp1 == 0 or temp2 == 0:
        return 1 * math.log(1, 2)
    entropy = - (temp1 * math.log(temp1, 2) + temp2 * math.log(temp2, 2))
    return entropy

def calc_label_entropy(matrix):
    N1 = np.count_nonzero(matrix[:, 0])
    N = matrix.shape[0]
    return calc_entropy(N1, N)

def calc_attr_entropy(result):
    appear_N = result[0]
    appear_N1 = result[1]
    non_appear_N = result[2]
    non_appear_N1 = result[3]

    entropy1 = calc_entropy(appear_N1, appear_N)
    entropy2 = calc_entropy(non_appear_N1, non_appear_N)

    total = appear_N + non_appear_N
    entropy = appear_N / total * entropy1 + non_appear_N / total * entropy2
    return entropy

def cal_info_gains(matrix, ranges):
    pqueue = Q.PriorityQueue()
    start = ranges[0]
    end = ranges[1]
    range_total = end - start + 1
    entropy = calc_label_entropy(matrix)
    for i in range(1, matrix.shape[1]):
        print(i)
        appear_N = np.count_nonzero(matrix[:, i])
        non_appear_N = matrix.shape[0] - appear_N
        appear_N1 = np.count_nonzero(matrix[start: end + 1, i])
        non_appear_N1 = range_total - appear_N1

        result = []
        result.append(appear_N)
        result.append(appear_N1)
        result.append(non_appear_N)
        result.append(non_appear_N1)

        attr_entropy = calc_attr_entropy(result)
        result.clear()
        info_gain = entropy - attr_entropy

        feature_value = FeatureValue(info_gain, i - 1)
        pqueue.put(feature_value)

    return pqueue

def get_top_100_feature(pqueue):
    count = 0
    features = []
    c = open("top_100_features.csv", "w")
    writer = csv.writer(c)
    while not pqueue.empty() and count < 100:
        feature_value = pqueue.get()
        attr_Id = getattr(feature_value, 'attr_Id')
        info_gain = getattr(feature_value, 'info_gain')
        writer.writerow([attr_Id, count, info_gain])
        features.append(attr_Id)
        count += 1
    print("finished")
    return features

def select_features(sample_file_name, label_file_name):
    labels = parse_labels(label_file_name)
    matrix = parse_samples(sample_file_name)
    matrix_with_label = cocat(labels, matrix)
    ranges = get_range(labels)
    pqueue = cal_info_gains(matrix_with_label, ranges)
    features = get_top_100_feature(pqueue)
    return features

def main():
    sample_file_name = 'matlab/train_data.csv'
    label_file_name = 'matlab/train_label.csv'
    select_features(sample_file_name, label_file_name)

if __name__ == '__main__':
   main()







