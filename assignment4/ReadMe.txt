1. Load the folder Assignment into the ide as an project, the project structure should be like this:
|-> Assigment
    |-> hw4
        |-> FeatureSelection.py   
        |-> GradientDescent.py
        |-> LogisticRegression.py
        |-> matlab(folder)

2. For question (a), run FeatureSelection.py and the program will output the selected features to the top_100_features.csv which will be used later under hw4.

3. For question (b), run GradientDescent.py and the program will output the weight vector to the weight_vector.csv which will be used later under hw4.
   You can change the learning rate in the main() function which is 0.001 now 
   You can change the iteration times in the run_gradient_descent() function which is 500 now

4. For question (c) (d) (f), run the LogisticRegression.py (which use the weight vector and selected features calculated before), and the program will output the accuracy, confusion matrix, recall and precision 

   For question (e), change the prediction threshold in the predict() function which is 0.5 now to get the answer.

————Notice————
1) We treat feature as the binary feature(present or not) during the feature selection
2) Term id and Attribute id start from 0


