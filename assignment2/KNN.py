import csv
import math
import operator

def load_csv(filename):
    lines = csv.reader(open(filename, "r"))
    dataset = list(lines)
    for i in range(len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]
    return dataset

def scale_train_data(list_of_attr):
    min_value = min(list_of_attr)
    max_value = max(list_of_attr)
    differ = float(max_value - min_value)
    for i in range(len(list_of_attr)):
        list_of_attr[i] = (list_of_attr[i] - min_value) / differ

    return list_of_attr

def scale_train_dataset(trainset):
    list_of_attr1 = [x[0] for x in trainset]
    list_of_attr2 = [x[1] for x in trainset]

    scaled_attr1 = scale_train_data(list_of_attr1)
    scaled_attr2 = scale_train_data(list_of_attr2)
    for i in range(len(trainset)):
        trainset[i][0] = scaled_attr1[i]
        trainset[i][1] = scaled_attr2[i]

def scale_test_data(list_of_attr, min_value, max_value):
    differ = float(max_value - min_value)
    for i in range(len(list_of_attr)):
        list_of_attr[i] = (list_of_attr[i] - min_value) / differ
    return list_of_attr

def scale_test_dataset(trainset, testset):
    list_of_train_attr1 = [x[0] for x in trainset]
    list_of_train_attr2 = [x[1] for x in trainset]

    min_1 = min(list_of_train_attr1)
    min_2 = min(list_of_train_attr2)
    max_1 = max(list_of_train_attr1)
    max_2 = max(list_of_train_attr2)

    list_of_test_attr1 = [x[0] for x in testset]
    list_of_test_attr2 = [x[1] for x in testset]

    scaled_attr1 = scale_test_data(list_of_test_attr1, min_1, max_1)
    scaled_attr2 = scale_test_data(list_of_test_attr2, min_2, max_2)

    for i in range(len(testset)):
        testset[i][0] = scaled_attr1[i]
        testset[i][1] = scaled_attr2[i]

def divide_train_dataset(dataset, index):
    end = len(dataset)
    if index == 0:
        return dataset[1:end]
    if index == end:
        return dataset[0:end - 1]
    return dataset[0:index] + dataset[index + 1: end]

def calc_eucDistance(data1, data2, length):
    dist = 0
    for i in range(length):
        dist += pow((data1[i] - data2[i]), 2)
    return math.sqrt(dist)

def get_k_neighboor_result(test_data, dataset, k):
    distances = []
    length = len(test_data) - 1
    for i in range(len(dataset)):
        dist = calc_eucDistance(test_data, dataset[i], length)
        distances.append((dataset[i], dist))
    distances.sort(key=operator.itemgetter(1))
    kResult = []
    for i in range(k):
        kResult.append(distances[i][0][2])
    return kResult

def predict(test_data, dataset, k):
    kResult = get_k_neighboor_result(test_data, dataset, k)
    number_of_0 = kResult.count(0)
    if number_of_0 > k/2:
        return float(0)
    return float(1)

def choose_k(dataset, k_candidates):
    chosen_k = 0
    max_accuracy = 0
    total = len(dataset)
    for k in k_candidates:
        count = 0
        for i in range(len(dataset)):
            actual = dataset[i][2]
            train_set = divide_train_dataset(dataset, i)
            prediction = predict(dataset[i], train_set, k)
            if prediction == actual:
                count += 1
        accuracy = count / total
        print("Choose k =", k, count)
        if accuracy > max_accuracy:
            chosen_k = k
            max_accuracy = accuracy
    return chosen_k


def predict_for_dataset(testset, trainset, k):
    tp_count = 0 #prediction 1, actual 1
    fp_count = 0 #prediction 1, actual 0
    tn_count = 0 #prediction 0, actual 0
    fn_count = 0 #prediction 0, actual 1
    total = len(testset)
    for i in range(total):
        test_data =testset[i]
        actual = test_data[2]
        prediction = predict(test_data, trainset, k)
        if actual == 1.0 and actual == prediction:
            tp_count += 1
        elif actual == 1.0 and actual != prediction:
            fn_count += 1
        elif actual == float(0) and actual == prediction:
            tn_count += 1
        elif actual == float(0) and actual != prediction:
            fp_count += 1
    accuracy = (tp_count + tn_count) / total
    print("Testset accuracy:", '{:.4%}'.format(accuracy))
    print("True positive rate:", tp_count/(tp_count + fn_count))
    print("False positive rate:", fp_count/(tn_count + fn_count))
    return [[tp_count, fn_count], [fp_count, tn_count]]

def main():
    k_candidates = [3, 9, 99]
    test_filename = 'banknote_test.csv'
    train_filename = 'banknote_train.csv'
    testset = load_csv(test_filename)
    trainset = load_csv(train_filename)
    scale_test_dataset(trainset, testset)
    scale_train_dataset(trainset)
    k = choose_k(trainset, k_candidates)
    confusion_matrix = predict_for_dataset(testset, trainset, k)
    print("Confusion Matrix:", confusion_matrix)

if __name__ == '__main__':
   main()



