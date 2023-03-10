import random as rand
import pandas as pd
import numpy as np
import math

def read_in_data():
    data_frame = pd.read_csv("small-test-dataset.txt",header = None, delim_whitespace=True)
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

def forward_selection(feature_vectors):
    current_feature_set = []
    max_accuracy = -1

    for i in range(1, len(feature_vectors)):
        curr_accuracy = 0
        considered_features = []

        for j in range(1, len(feature_vectors)):
            
            if j not in current_feature_set:
                temp_curr_set = current_feature_set.copy()
                temp_curr_set.append(j)
                curr_accuracy = leave_one_out_cross_validation(temp_curr_set)
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

def backward_elimination(feature_vectors):
    current_feature_set = feature_vectors.copy() #dont wanna change feature_vectors
    max_accuracy = -1
    current_feature_set.pop(0)

    for i in range(0, len(feature_vectors)):
        curr_accuracy = 0
        considered_features_to_remove = []

        for j in range (0, len(feature_vectors)):

            if j in current_feature_set:
                
                temp_curr_set = current_feature_set.copy()
                temp_curr_set.remove(j)
                curr_accuracy = leave_one_out_cross_validation(temp_curr_set)
                temp_curr_set_accuracy_tuple = (temp_curr_set, curr_accuracy)
                considered_features_to_remove.append(temp_curr_set_accuracy_tuple)

                print("Using feature(s)", temp_curr_set, "accuracy is", curr_accuracy)
        
        candidate_feature = get_max_accuracy_tuple(considered_features_to_remove)

        print("\n")

        if candidate_feature[1] > max_accuracy:
            print("Feature set", candidate_feature[0], "was best, accuracy is", candidate_feature[1], "\n")
            max_accuracy = candidate_feature[1]
            current_feature_set = candidate_feature[0].copy()
        
        else:
            break


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

def knn_classifier(training_data, test_data, training_label, num_neighbors):
    '''
    arguments:
        test_data (1d numpy array) : singular test point
        training data(2d numpy array) : each row is a training point
        training_labels (1d numpy array): each index is a row's training label
        num_neighbors(int): number of neighbors

    returns:
        prediction label
    
    '''
    distance_array = []
    
    #for i in range(len(training_data)):
    distance_to_test_data = euclidian_distance(training_data, test_data)
    distance_training_label_tuple = (distance_to_test_data, training_label)

    distance_array.append(distance_training_label_tuple)

    distance_array = sorted(distance_array)

    predicted_label_array = []
    for i in range(num_neighbors):
        predicted_label_array.append(distance_array[i])

    return predicted_label_array

def euclidian_distance(point1, point2):
    #point1 numpy training point, point2 numpy test point
    print("training point:", point1)
    distance = 0

    for i in range(len(point1)):
        distance += (point1[i] - point2[i])**2
    
    distance = math.sqrt(distance)

    print("distance to test point:", distance)
    return distance

def leave_one_out_cross_validation(data_set, considered_feature_set):
    number_correctly_classified = 0
    considered_feature_set_values = []
    data_set_labels = data_set[0].values
    
    # for i in range(len(considered_feature_set)):
    #     considered_feature_set_values.append(data_set[considered_feature_set[i]])
    
    # considered_feature_set_values = np.array(considered_feature_set_values)

    considered_feature_set_values = data_set[considered_feature_set].values
    print(considered_feature_set_values[1])

    for i in range(len(considered_feature_set_values)):
        object_to_classify = considered_feature_set_values[i]
        object_to_classify_label = data_set_labels[i]

        for k in range (len(considered_feature_set_values)):
            
            if k != i:
                predicted_label = knn_classifier(considered_feature_set_values[k], object_to_classify,data_set_labels[k], 1)

        if object_to_classify_label == predicted_label[0][1]:
            number_correctly_classified += 1
    
    accuracy = number_correctly_classified / len(considered_feature_set_values)
    return accuracy

def intialize_feature_set(data_frame):
    starting_feature_set = data_frame[0:].values
    return starting_feature_set


def main():
    #int_feature_count, int_algo_choice = get_user_input()
    data_frame = read_in_data()

    print("starting_feature_set:")
    starting_feature_set = intialize_feature_set(data_frame)

    print(starting_feature_set)

    # temporary_data_list = [0,1,2,3,4,5,6,7,8,9]
    # backward_elimination(temporary_data_list)

    # labels = data_frame[0]
    # labels = np.array(labels)

    # point_1 = [[39.1, 18.7, 181.0],
    #            [39.5, 17.4, 186.0],
    #            [47.2, 13.7, 214.0],
    #            [50.4, 15.7, 222.0]]
    
    # point_2 = [39.3, 20.6, 190.0]

    # training_labels = [0,0,1,1]

    # point_1 = np.array(point_1)
    # point_2 = np.array(point_2)

    # # distance = euclidian_distance(point_1, point_2)

    # predicted_label = knn_classifier(point_1, point_2, training_labels, 1)
    # print("")
    # print(predicted_label)
    # considered_feature_set = [1, 2, 3]
    # print("ACCURACY:", accuracy)



    #print("Feature Count:", int_feature_count, ",", "Algorithm Choice", int_algo_choice)





main()


