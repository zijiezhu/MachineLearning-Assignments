from assignment4.FeatureSelection import *

def get_features(feature_file_name):
    map = {}
    lines = csv.reader(open(feature_file_name, "r"))
    features = list(lines)
    for i in range(len(features)):
        key = int(features[i][0])
        value = int(features[i][1])
        map[key] = value
    return map

def generate_matrix(sample_file_name, label_file_name, features, row_num):
    raw_labels = list(csv.reader(open(label_file_name, "r")))
    raw_samples = list(csv.reader(open(sample_file_name, "r")))
    row = []
    col = []
    data = []
    binary_labels = []
    for i in range(len(raw_samples)):
        sample = raw_samples[i][0].split(" ")
        cur_col = int(sample[1]) - 1  # attrId start from 0
        if cur_col not in features:
            continue
        cur_row = int(sample[0]) - 1  # docId start from 0
        cur_data = int(sample[2])
        row.append(cur_row)
        col.append(features.get(cur_col))  # attrId map to 0 - 99
        data.append(cur_data)

    for i in range(len(raw_labels)):
        label = int(raw_labels[i][0])
        if label >= 12 and label <= 15:
            binary_labels.append(1)
        else:
            binary_labels.append(0)

    matrix = sparse.coo_matrix((data, (row, col)), shape=(row_num, 100)).toarray()

    matrix_with_label = np.c_[binary_labels, matrix]

    return matrix_with_label

def sigmoid(weights, weight_0, row):
    row_without_label = row[1:]
    o = - (weights.dot(row_without_label) + weight_0)
    return 1.0 / (1 + np.power(np.e, o))

def run_gradient_descent(matrix, learning_rate):
    weights = np.random.uniform(-0.01, 0.01, matrix.shape[1] - 1)
    weight_0 = np.random.uniform(-0.01, 0.01)
    count = 0
    while count < 500:
        print(count)
        weights_delta = 0.0
        weight_delta_0 = 0.0
        for doc_id in range(matrix.shape[0]):
            y = sigmoid(weights, weight_0, matrix[doc_id])
            r = matrix[doc_id][0]
            row = matrix[doc_id]
            weight_delta_0 += r - y
            weights_delta += (r - y) * row[1:]

        weight_0 += learning_rate * weight_delta_0
        weights += learning_rate * weights_delta
        count += 1
    result = [weight_0, weights]
    return result

def main():
    sample_file_name = 'matlab/train_data.csv'
    label_file_name = 'matlab/train_label.csv'

    feature_file_name = 'top_100_features.csv'
    features = get_features(feature_file_name)
    matrix = generate_matrix(sample_file_name, label_file_name, features, 11269)
    w = run_gradient_descent(matrix, 0.001)
    c = open("weight_vector.csv", "w")
    writer = csv.writer(c)
    writer.writerow([w[0]])
    for i in range(len(w[1])):
        cur_w = w[1][i]
        writer.writerow([cur_w])

if __name__ == '__main__':
   main()