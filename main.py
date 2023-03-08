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
    return rand.randint(0,100)
#
# feature vectors is a 2d vector where outer vector is each feature
# and the inner vectors are the values of the features

def feature_search(feature_vectors): 
    current_feature_set = []
    curr_best_accr = -1

    for feature in range(1, len(feature_vectors)):
        print("On the", feature, "level of the search tree")
        feature_to_add_at_level = []
        accuracy = 0
        # curr_best_accr = -1

        for considered_feature in range(1, len(feature_vectors)):

            if considered_feature not in current_feature_set:
                print('\t--considering adding the', considered_feature, 'feature')
                accuracy = random_accuracy(current_feature_set)

                if accuracy > curr_best_accr:
                    curr_best_accr = accuracy
                    feature_to_add_at_level = considered_feature
                    current_feature_set.append(feature_to_add_at_level)

                    print("On level", feature, 'i added feature', feature_to_add_at_level, 'to current set')    
                    
                    print('best accuracy vector:', curr_best_accr)
                    print(current_feature_set)
                    print("\n")
                    
        # current_feature_set.append(feature_to_add_at_level)
        # print("On level", feature, 'i added feature', feature_to_add_at_level, 'to current set')       

    print('Current Feature set:',current_feature_set)     
    print('best accuracy:', curr_best_accr)

def new_feature_search(feature_vectors):
    current_feature_set = []
    max_accuracy = -1

    for i in range(1, len(feature_vectors)):
        curr_accuracy = 0
        considered_features = []

        for j in range(1, len(feature_vectors)):
            
            if j not in current_feature_set:
                print(current_feature_set)
                temp_curr_set = current_feature_set.copy()
                temp_curr_set.append(j)
                curr_accuracy = random_accuracy(temp_curr_set)
                temp_curr_set_accuracy_tuple = (temp_curr_set, curr_accuracy)
                considered_features.append(temp_curr_set_accuracy_tuple)

        
        candidate_feature = get_max_accuracy_tuple(considered_features)
       # print(candidate_feature)

        if candidate_feature[1] > max_accuracy:
            max_accuracy = candidate_feature[1]
            current_feature_set = candidate_feature[0].copy()

    print("Ending feature set:", current_feature_set)
    print("Ending max accuracy:", max_accuracy)

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

                    


def main():
    #int_feature_count, int_algo_choice = get_user_input()
    data_frame = read_in_data()

    temporary_data_list = [0,1,2,3,4,5,6,7,8,9]
    new_feature_search(temporary_data_list)


    #print("Feature Count:", int_feature_count, ",", "Algorithm Choice", int_algo_choice)





main()


