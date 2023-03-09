import random as rand
import pandas as pd

def read_in_data():
    data_frame = pd.read_csv("small-test-dataset.txt",header = None,sep='  ')
    print(data_frame)

    return data_frame

def get_user_input():
    print("Welcome to Alexander Kaattari-Lim, 862161616 Feature Selection Algorithm.\n")
    int_feature_count = int(input("Please enter total number of features: "))

    print("Type the number of the algorithm you want to run.")
    print("\t1. Forward Selection")
    print("\t2. Backward Elimination\n")

    int_algo_choice = int(input("Choice of algorithm:"))
    print("\n")

    while int_algo_choice != 1 and int_algo_choice != 2:
        
        print("Please choose a valid option")
        int_algo_choice = int(input("Choice of algorithm:"))


    return int_feature_count, int_algo_choice

# def leave_one_out_cross_validation(data, current_set_of_features, k+1):
#     return random_accuracy()

def random_accuracy(current_feature_set):
    return rand.uniform(0.0,100.0)
#
# feature vectors is a 2d vector where outer vector is each feature
# and the inner vectors are the values of the features

def new_feature_search(feature_vectors):
    current_feature_set = []
    max_accuracy = -1

    for i in range(1, len(feature_vectors)):
        curr_accuracy = 0
        considered_features = []

        for j in range(1, len(feature_vectors)):
            
            if j not in current_feature_set:
                temp_curr_set = current_feature_set.copy()
                temp_curr_set.append(j)
                curr_accuracy = random_accuracy(temp_curr_set)
                temp_curr_set_accuracy_tuple = (temp_curr_set, curr_accuracy)
                considered_features.append(temp_curr_set_accuracy_tuple)

                print("Using feature(s)", temp_curr_set, "accuracy is", curr_accuracy)

        
        candidate_feature = get_max_accuracy_tuple(considered_features)
       # print(candidate_feature)

        if candidate_feature[1] > max_accuracy:
            print("\nFeature set", candidate_feature[0], "was best, accuracy is", candidate_feature[1], "\n")
            max_accuracy = candidate_feature[1]
            current_feature_set = candidate_feature[0].copy()

    print("\nFinished search!! The best feature subset is", current_feature_set,"which has an accuracy of", max_accuracy)

def get_max_accuracy_tuple(tuple_list):
    curr_max = -1
    curr_max_index = -1

    for i in range(len(tuple_list)):

        if tuple_list[i][1] > curr_max:
            curr_max = tuple_list[i][1]
            curr_max_index = i

    if curr_max == -1:
        return (-1,-1)
    else:
        return tuple_list[curr_max_index]

def knn_classifier(training_data, test_data, training_labels, num_neighbors):
    pass             

def euclidian_distance(point1, point2):
    pass

def main():
    #int_feature_count, int_algo_choice = get_user_input()
    data_frame = read_in_data()

    temporary_data_list = [0,1,2,3,4,5,6,7,8,9]
    new_feature_search(temporary_data_list)


    #print("Feature Count:", int_feature_count, ",", "Algorithm Choice", int_algo_choice)





main()


