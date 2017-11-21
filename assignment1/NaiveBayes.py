import csv
import math

SMOOTH = 0.9

def load_csv(filename):
    lines = csv.reader(open(filename, "r"))
    dataset = list(lines)
    for i in range(len(dataset)):
        dataset[i] = [int(x) for x in dataset[i]]
    return dataset

def seperate_by_class(dataset):
    separated = {}

    for i in range(len(dataset)):
        cur = dataset[i]
        if (cur[-1] not in separated):
            separated[cur[-1]] = []
        separated[cur[-1]].append(cur)

    return separated

def calc_probability_of_class(separated):
    ligitimate_amount = len(separated.get(1))
    suspicious_amount = len(separated.get(0))
    phishing_amount = len(separated.get(-1))
    total = ligitimate_amount + suspicious_amount + phishing_amount

    class_probabilities = [phishing_amount/total, suspicious_amount/total, ligitimate_amount/total]
    return class_probabilities

def calc_probability_of_attr_with_value(separated, attr_probability, index, value, t):
    global SMOOTH
    list = []
    for i in range(3):
        cur_class = separated.get(i-1)
        total = len(cur_class)
        count = 0
        for j in range(total):
            cur = cur_class[j]
            if cur[index] == value:
                count += 1
        list.append((count + SMOOTH) /(total + SMOOTH * t))
        attr_probability[value] = list

def calc_probability_of_attr_has_3_values(separated, index):
    attr_probability = {}
    for value in range (-1, 2):
        calc_probability_of_attr_with_value(separated, attr_probability, index, value, 3)
    return attr_probability

def calc_probability_of_age_of_domain(separated, index):
    attr_probability = {}
    calc_probability_of_attr_with_value(separated, attr_probability, index, -1, 2)
    calc_probability_of_attr_with_value(separated, attr_probability, index, 1, 2)
    return attr_probability

def calc_probability_of_having_ip(separated, index):
    attr_probability = {}
    calc_probability_of_attr_with_value(separated, attr_probability, index, 0, 2)
    calc_probability_of_attr_with_value(separated, attr_probability, index, 1, 2)
    return attr_probability

def calc_probability_of_attrs(separated):
    attr_probabilities = {}
    for i in range(7):
        attr_probability = calc_probability_of_attr_has_3_values(separated, i)
        attr_probabilities[i] = attr_probability

    age_of_domain_probability = calc_probability_of_age_of_domain(separated, 7)
    attr_probabilities[7] = age_of_domain_probability

    having_ip_probability = calc_probability_of_having_ip(separated, 8)
    attr_probabilities[8] = having_ip_probability

    return attr_probabilities

def predict(class_probabilities, attr_probabilities, row):
    max_likelihood = float('-inf')
    prediction = 0
    for i in range(3):
        likelihood = math.log(class_probabilities[i])
        for j in range(9):
            attr_probability = attr_probabilities.get(j).get(row[j])
            likelihood += math.log(attr_probability[i])
        if likelihood > max_likelihood:
            max_likelihood = likelihood
            prediction = i - 1
    return prediction

def calc_accuracy(class_probabilities, attr_probabilities, dataset):
    hit = 0
    total = len(dataset)

    for i in range(total):
        cur = dataset[i]
        prediction = predict(class_probabilities, attr_probabilities, cur)
        if prediction == cur[-1]:
            hit += 1
    return hit/total

def calc_accuracy_by_zeroR(dataset):
    hit = 0
    total = len(dataset)
    for i in range(total):
        cur = dataset[i]
        if -1 == cur[-1]:
            hit += 1
    return hit/total

def main():
    train_name = 'phish_train.csv'
    test_name = 'phish_test.csv'

    test_data = load_csv(test_name)
    train_data = load_csv(train_name)

    separated = seperate_by_class(train_data)

    class_probabilities = calc_probability_of_class(separated)
    attr_probabilities = calc_probability_of_attrs(separated)

    train_accuracy = calc_accuracy(class_probabilities, attr_probabilities, train_data)
    test_accuracy = calc_accuracy(class_probabilities, attr_probabilities, test_data)

    train_accuracy_zeroR = calc_accuracy_by_zeroR(train_data)
    test_accuracy_zeroR = calc_accuracy_by_zeroR(test_data)

    print('Train accuracy: ', train_accuracy)
    print('Test accuracy: ', test_accuracy)
    print('Attribute probabilities: ', attr_probabilities)
    print('Class probabilities: ', class_probabilities)
    print('Train accuracy by zeroR:', train_accuracy_zeroR)
    print('Test accuracy by zeroR:', test_accuracy_zeroR)

if __name__ == '__main__':
   main()
