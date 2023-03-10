import random as rand
import pandas as pd
import numpy as np
import math

def read_in_data():
    data_frame = pd.read_csv("CS170_Spring_2022_Small_data__35.txt",header = None, delim_whitespace=True)
    #print(data_frame)

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

def forward_selection(data_frame, num_columns):
    current_feature_set = []
    max_accuracy = -1

    for i in range(1, num_columns):
        curr_accuracy = 0
        considered_features = []

        for j in range(1, num_columns):
            
            if j not in current_feature_set:
                temp_curr_set = current_feature_set.copy()
                temp_curr_set.append(j)
                curr_accuracy = leave_one_out(data_frame, temp_curr_set)
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

# def map_feature_idx_to_feature_vals(all_feature_data, curr_feature_set_values, curr_feature_set_idxs):
#     new_curr_feature_set_values = []

#     for i in range(len(curr_feature_set_idxs)):
#         new_curr_feature_set_values.append()
    


def backward_elimination(data_frame,num_columns):

    current_feature_set = []

    for i in range(1,num_columns):
        current_feature_set.append(i)

    max_accuracy = -1

    for i in range(1, num_columns):
        curr_accuracy = 0
        considered_features_to_remove = []

        for j in range (1, num_columns):

            if j in current_feature_set:
                
                temp_curr_set = current_feature_set.copy()
                temp_curr_set.remove(j)
                curr_accuracy = leave_one_out(data_frame, temp_curr_set)
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
    #print("training point:", point1)
    distance = 0

    for i in range(len(point1)):
        distance += (point1[i] - point2[i])**2
    
    distance = math.sqrt(distance)

    return distance

def leave_one_out_cross_validation(data_set, considered_feature_set):
    number_correctly_classified = 0
    considered_feature_set_values = []
    data_set_labels = data_set[0].values
    
    # for i in range(len(considered_feature_set)):
    #     considered_feature_set_values.append(data_set[considered_feature_set[i]])
    
    # considered_feature_set_values = np.array(considered_feature_set_values)

    considered_feature_set_values = data_set[considered_feature_set].values
    print(considered_feature_set_values)

    for i in range(len(considered_feature_set_values)):
        object_to_classify = considered_feature_set_values[i]
        object_to_classify_label = data_set_labels[i]

        for k in range (len(considered_feature_set_values)):
            
            if k != i:
                predicted_label = knn_classifier(considered_feature_set_values[k], object_to_classify,data_set_labels[k], 1)

        if object_to_classify_label == predicted_label[0][1]:
            number_correctly_classified += 1
    
    accuracy = number_correctly_classified / len(data_set)
    return accuracy

def leave_one_out(data_set, considered_feature_set):

    number_correctly_classified = 0
    considered_feature_set_values = []
    considered_feature_set_values = data_set[considered_feature_set].values
    data_set_labels = data_set[0].values

    for i in range(len(considered_feature_set_values)):
        object_to_classify = considered_feature_set_values[i]
        object_to_classify_label = data_set_labels[i]
        nn_neighbor_label = -1
        
        nn_dist = 999999
        nn_location = 999999

        for k in range(len(considered_feature_set_values)):
                
            if k != i:

                distance = euclidian_distance(object_to_classify, considered_feature_set_values[k])
                if distance < nn_dist:
                    nn_dist = distance
                    nn_location = k
                    nn_neighbor_label = data_set_labels[k]

        if object_to_classify_label == nn_neighbor_label:
            number_correctly_classified += 1


    accuracy = number_correctly_classified / len(data_set_labels)
    return accuracy * 100

def feature_normalization(data_frame):
    normalized_data_frame = data_frame.copy()

    for column in range(1, len(normalized_data_frame.columns)):
        normalized_data_frame[column] = (normalized_data_frame[column] - normalized_data_frame[column].mean()) / normalized_data_frame[column].std()

    return normalized_data_frame

def open_file():
    file_handle = open("results.txt", "a")
    return file_handle

def main():
    #int_feature_count, int_algo_choice = get_user_input()
    data_frame = read_in_data()
    #print(data_frame)
    data_frame = feature_normalization(data_frame)
    print(data_frame)

    #backward_elimination(data_frame, len(data_frame.columns))
    forward_selection(data_frame, len(data_frame.columns))


    # label_values = data_frame[0].values
    # twos_count = 0
    # ones_count = 0

    # for i in label_values:
    #     if i == 2:
    #         twos_count += 1
    #     elif i == 1:
    #         ones_count += 1
    
    # accuracy = twos_count / len(label_values)
    # print(accuracy)

    #print(len(data_frame.columns))
    #forward_selection(data_frame, len(data_frame.columns))

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


