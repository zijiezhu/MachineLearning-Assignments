from assignment4.GradientDescent import *

# True Positive Rate : True Positive results / results that were supposed to be positive ( r = +, y = + ) / (r = +)
# False Positive Rate: False Positive results / results that were supposed to be negative ( r = -, y = + ) / (r = -)
# True Negative: (r = -, y = -) / (r = -)
# False Negative: (r = +, y = -) / (r = +)
# Precision: TP / (TP + FP) = (r = +, y = +) / (r = +, y = +) (r = -, y = +)

def get_weight_vector(vector_file_name):
    weight_vector = []
    lines = csv.reader(open(vector_file_name, "r"))
    dataset = list(lines)
    for i in range(len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]
        weight_vector.append(dataset[i][0])
    return weight_vector

def predict(vector, data):
    expo = vector[0]
    for i in range(1, len(data)):
        expo += vector[i] * data[i]
    try:
        prediction = 1 / (1 + math.exp(- expo))
        if prediction >= 0.5:
            return 1
        else:
            return 0
    except OverflowError:
        return 0

def predict_for_dataset(matrix, vector):
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    hit = 0
    for i in range(matrix.shape[0]):
        data = matrix[i, :]
        prediction = predict(vector, data)
        actual = matrix[i][0]
        if (prediction == actual):
            hit += 1
            if (prediction == 1):
                true_positive += 1
            else:
                true_negative += 1
        elif prediction == 1:
            false_positive += 1
        else:
            false_negative += 1

    total = matrix.shape[0]
    pos_total = np.count_nonzero(matrix[:, 0])
    neg_total = total - pos_total
    print("true positive: ", true_positive)
    print("true negative: ", true_negative)
    print("false positive: ", false_positive)
    print("false negative: ", false_negative)
    print("Recall: ", true_positive / pos_total)
    print("Precision: ", true_positive / (true_positive + false_positive))
    print("Accuracy: ", hit / total)
    result = [true_positive / pos_total, true_negative / neg_total, false_positive / neg_total, false_negative / pos_total, true_positive / (true_positive + false_positive), hit / total]
    return result

def main():
    vector_file = 'weight_vector.csv'
    feature_file = 'top_100_features.csv'
    test_sample_file = 'matlab/test_data.csv'
    test_label_file = 'matlab/test_label.csv'
    vector = get_weight_vector(vector_file)
    features = get_features(feature_file)
    matrix = generate_matrix(test_sample_file, test_label_file, features, 7505)
    result = predict_for_dataset(matrix, vector)
    print(result)

if __name__ == '__main__':
   main()
