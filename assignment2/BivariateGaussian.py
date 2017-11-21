import math

import numpy as np

from assignment2 import KNN

def seperate_by_label(dataset, label):
    seperated = []
    for i in range(len(dataset)):
        data = dataset[i]
        if data[2] == label:
            seperated.append(data)
    return seperated

def calc_estimates(seperated):
    total1 = 0
    total2 = 0
    n = len(seperated)
    for i in range(len(seperated)):
        data = seperated[i]
        total1 += data[0]
        total2 += data[1]
    return [total1/n, total2/n]

def calc_covariance_matrix_item(attr1, e1, attr2, e2):
    n = len(attr1)
    total = 0
    for i in range(n):
        value1 = attr1[i]
        value2 = attr2[i]
        total += (value1 - e1) * (value2 - e2)
    return float(total / n)

def calc_covariance_matrix(dataset, estimates):
    attr1 = [x[0] for x in dataset]
    attr2 = [x[1] for x in dataset]

    item1 = calc_covariance_matrix_item(attr1, estimates[0], attr1, estimates[0])
    item2 = calc_covariance_matrix_item(attr1, estimates[0], attr2, estimates[1])
    item3 = calc_covariance_matrix_item(attr2, estimates[1], attr2, estimates[1])

    return np.matrix([[item1, item2],[item2, item3]])

def calc_pdf(data, estimates, matrix):
    data = np.matrix(data)
    estimates = np.matrix(estimates)
    matrix_det = np.linalg.det(matrix)
    matrix_inv = np.linalg.inv(matrix)
    temp = data - estimates
    pdf = -1 * math.log(2 * 3.14) - 1/2 * math.log(matrix_det) - 1/2 * temp * matrix_inv * temp.T
    return float(pdf)

def predict_for_dataset(testset, estimates1, matrix1, estimates2, matrix2):
    estimates1 = np.matrix(estimates1)
    estimates2 = np.matrix(estimates2)
    tp_count = 0  # prediction 1, actual 1
    fp_count = 0  # prediction 1, actual 0
    tn_count = 0  # prediction 0, actual 0
    fn_count = 0  # prediction 0, actual 1
    total = len(testset)
    for i in range(total):
        test_data = testset[i]
        actual = test_data[2]
        pdf1 = calc_pdf(test_data[:2], estimates1, matrix1)
        pdf2 = calc_pdf(test_data[:2], estimates2, matrix2)
        if pdf1 > pdf2:
            prediction = 1
        else:
            prediction = 0

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
    print("True positive rate:", tp_count / (tp_count + fn_count))
    print("False positive rate:", fp_count / (fp_count + tn_count))
    return [[tp_count, fn_count], [fp_count, tn_count]]

def main():

    testset = KNN.load_csv('banknote_test.csv')
    trainset = KNN.load_csv('banknote_train.csv')

    data_of_label1 = seperate_by_label(trainset, 1)
    data_of_label0 = seperate_by_label(trainset, 0)
    estimates_of_label1 = calc_estimates(data_of_label1)
    estimates_of_label0 = calc_estimates(data_of_label0)

    matrix_of_label1 = calc_covariance_matrix(data_of_label1, estimates_of_label1)
    matrix_of_label0 = calc_covariance_matrix(data_of_label0, estimates_of_label0)

    print("estimates_of_label1:", estimates_of_label1)
    print("estimates_of_label0:", estimates_of_label0)
    print("matrix_of_label1", matrix_of_label1)
    print("matrix_of_label0", matrix_of_label0)

    confusion_matrix = predict_for_dataset(testset, estimates_of_label1, matrix_of_label1, estimates_of_label0,matrix_of_label0)
    print("Confusion Matrix", confusion_matrix)

if __name__ == '__main__':
   main()
