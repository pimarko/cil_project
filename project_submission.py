#import libraries
import numpy as np
import csv
from datetime import datetime
import matplotlib.pyplot as plt
import sys
import random
import os.path
import matplotlib

"""
GENERATE_SUBMISSION:
Set to True to run the final optimization with the set BEST_K and generate the submission file uploaded on Kaggle.

GRID_SEARCH:
Set to true if still in search for the optimal number of concepts in our data. Produces a plot and the code exits right after.

K:
Number of concepts to check in the grid search for low rank approximation of the data matrix using the incremental SVD. 
Max. K is 1000, as we have data matrix of rank 1000.

BEST_K:
Set the optimal number of concepts in the data according to the grid search plot obtained and set GENERATE_SUBMISSION to True.

NMB_OF_TRAINING_ITERATIONS:
Number of iterations is set, but the algorithm checks how the training and validation error behave. If the validation error starts to grow, it aborts. 
If the training error achieves EPS error, it aborts (also provide sanity check that training error decreases).

SEED_NUM:
Seed num set to avoid stochastic nature of the optimization.

LEARNING RATE:
Learning rate is set fixed to 0.001. Due to potential overshooting of the minimum or too slow convergence, it might be dynamically adapted. We observe the
validation error and well as training error behaviour to be ably to abort early. 

REGULARIZATION_TERM:
Set to optimal one obtained after the grid search.

REG_TERMS:
Set to number of regularization terms to be tested.
"""

#constants to be adapted
SUBMISSION_FILENAME = "my_submission_final.csv"
GENERATE_SUBMISSION = False

#optimal parameters
BEST_K = 10

#training
GRID_SEARCH = True
K = 150
LEARNING_RATE = 0.001
NMB_OF_TRAINING_ITERATIONS = 20000000
SEED_NUM = 500
REGULARIZATION_TERM = 0
REG_TERMS = 15
EPS = 0.1


#calculate the loss function
def irmse(predicted_matrix,validation_ids):
    num_of_items = len(validation_ids)
    error = 0

    for validation_id in validation_ids:
        item = validation_id[0]
        item_index = item[0]
        user_index = item[1]
        item_rating = item[2]

        error = error + (np.square(predicted_matrix[item_index,user_index]-item_rating))

    return np.sqrt(error/num_of_items)

#claculate the sgd 
def sgd(x_dn,u_d,z_n,stepsize, reg_term):
    dotProd = np.dot(u_d,z_n)

    grad_u_d = -(x_dn-dotProd)*z_n + 2*reg_term*u_d
    if (np.any(np.isnan(grad_u_d))):
        print x_dn, u_d, z_n
        sys.exit()

    grad_z_n = -(x_dn-dotProd)*u_d + 2*reg_term*z_n

    u_d = u_d - stepsize*grad_u_d
    z_n = z_n - stepsize*grad_z_n

    return u_d,z_n

#set the seed to get determinism
np.random.seed(SEED_NUM)

#check if data already provided
if (not os.path.isfile("training_ids.npy") or not os.path.isfile("validation_ids.npy") or not os.path.isfile("test_ids.npy")):
    train_ids = []
    train_predictions = []
    with open("data_train.csv","rb") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            train_ids.append(row['Id'])
            train_predictions.append(row['Prediction'])


    test_ids = []
    test_predictions = []
    with open("sampleSubmission.csv","rb") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            test_ids.append(row['Id'])
            test_predictions.append(row['Prediction'])


    print "Data loading done."


    #build data matrix and split the data into training(80% of data) and validation set(20% of data)
    validation_ids = []
    training_ids = []
    train_matrix = np.zeros((1000,10000))

    for k in range(0,len(train_ids)):
        train_id = train_ids[k]

        index_r = train_id.index('r')
        index_c = train_id.index('c')
        length = len(train_id)

        i = int(train_id[index_c+1:length])-1
        j = int(train_id[index_r+1:index_c-1])-1


        rand_num = random.random()
        if(rand_num < 0.2):
            validation_ids.append([(i,j,int(train_predictions[k]))])
        else:
            training_ids.append([(i,j,int(train_predictions[k]))])
            train_matrix[i,j] = int(train_predictions[k])

    print "Data matrix built."
    print "Number of items in the validation set: " + str(len(validation_ids)) + "."
    print "Number of items in the training set: " + str(np.count_nonzero(train_matrix)) + "."

    np.save("training_ids", training_ids)
    np.save("validation_ids", validation_ids)
    np.save("test_ids", test_ids)

else:
    training_ids = np.load("training_ids.npy")
    validation_ids = np.load("validation_ids.npy")
    test_ids = np.load("test_ids.npy")

    print "Data loaded from files."

#prepare random draws from data set
rand_ids = np.random.choice(range(0,len(training_ids)), size=NMB_OF_TRAINING_ITERATIONS)

#grid search for optimal hyperparameteres k and reg. term using sgd algorithm
if(GRID_SEARCH):
    k_search = np.linspace(1,K,K).astype(int)
    reg_terms = np.linspace(0,1,REG_TERMS)
    rmse = np.zeros((K,REG_TERMS))
    for reg_term in range(len(reg_terms)):
        for k in range(len(k_search)):
            U = np.random.rand(1000,k_search[k])
            Z = np.random.rand(10000,k_search[k])
            j = 1
            validate_err_curr = np.inf
            validate_err_prev = np.inf
            training_err_curr = np.inf
            training_err_prev = np.inf
            for rand_idx in range(len(rand_ids)):
                training_id = training_ids[rand_ids[rand_idx]]
                nz_item = training_id[0]
                d = nz_item[0]
                n = nz_item[1]
                rating = nz_item[2]
                
                U[d,:],Z[n,:] = sgd(rating,U[d,:],Z[n,:],LEARNING_RATE, reg_terms[reg_term])


                if (np.mod(j,500000) == 0 or j == 1):
                    prediction_matrix = np.dot(U,Z.T)
                    validate_err_prev = validate_err_curr
                    validate_err_curr = irmse(prediction_matrix,validation_ids)
                    training_err_prev = training_err_curr
                    training_err_curr = irmse(prediction_matrix,training_ids)
                    if(validate_err_prev  < validate_err_curr or training_err_prev < training_err_curr or training_err_curr < EPS):
                        print "Early abort."
                        print "Current validation error: " + str(validate_err_curr) + "."
                        print "Previous validation error: " + str(validate_err_prev) + "."

                        print "Current training error: " + str(training_err_curr) + "."
                        print "Previous training error: " + str(training_err_prev) + "."
                        break
                    else:
                        print "At iteration: " + str(j) + "."

                j = j + 1

            prediction_matrix = np.dot(U,Z.T)
            rmse[k,reg_term] = irmse(prediction_matrix,validation_ids)
            print "Optimization done."


    colors = []
    for name, hex in matplotlib.colors.cnames.iteritems():
        colors.append(str(name))

    for reg_term in range(0,len(reg_terms)):
        labeling = 'alpha ' + str(reg_terms[reg_term]) 
        plt.plot(k_search, rmse[:,reg_term],c = colors[reg_term],label=labeling)

    plt.xlabel('K features selected')
    plt.ylabel('RMSE')
    plt.title('Feature selection based on the RMSE using incremental SVD')
    plt.show()
    sys.exit()

#prediction
if(GENERATE_SUBMISSION):
    #run the optimization
    U = np.random.rand(1000,BEST_K)
    Z = np.random.rand(10000,BEST_K)
    j = 1
    validate_err_curr = np.inf
    validate_err_prev = np.inf
    training_err_curr = np.inf
    training_err_prev = np.inf
    for rand_idx in range(len(rand_ids)):
        training_id = training_ids[rand_ids[rand_idx]]
        nz_item = training_id[0]
        d = nz_item[0]
        n = nz_item[1]
        rating = nz_item[2]
        
        U[d,:],Z[n,:] = sgd(rating,U[d,:],Z[n,:],LEARNING_RATE, REGULARIZATION_TERM)


        if (np.mod(j,500000) == 0 or j == 1):
            prediction_matrix = np.dot(U,Z.T)
            validate_err_prev = validate_err_curr
            validate_err_curr = irmse(prediction_matrix,validation_ids)
            training_err_prev = training_err_curr
            training_err_curr = irmse(prediction_matrix,training_ids)
            if(validate_err_prev  < validate_err_curr or training_err_prev < training_err_curr or training_err_curr < EPS):
                print "Early abort."
                print "Current validation error: " + str(validate_err_curr) + "."
                print "Previous validation error: " + str(validate_err_prev) + "."

                print "Current training error: " + str(training_err_curr) + "."
                print "Previous training error: " + str(training_err_prev) + "."
                break
            else:
                print "At iteration: " + str(j) + "."

        j = j + 1

    prediction_matrix = np.dot(U,Z.T)
    print "Optimization done."

    #prediction preparation
    item_predict_index = []
    user_predict_index = []
    testing_ids = []
    for k in range(0,len(test_ids)):
        test_id = test_ids[k]

        index_r = test_id.index('r')
        index_c = test_id.index('c')
        length = len(test_id)

        i = int(test_id[index_c+1:length])-1
        j = int(test_id[index_r+1:index_c-1])-1

        item_predict_index.append(i)
        user_predict_index.append(j)
        testing_ids.append([(i,j)])

    prediction_indices = zip(item_predict_index,user_predict_index)

    predictions = []
    for prediction_index in prediction_indices:
        predictions.append(prediction_matrix[prediction_index[0],prediction_index[1]])

    print "Predictions ready."

    #generate prediction file
    predictions_file = open("submissions/" + SUBMISSION_FILENAME, "wb")
    open_file_object = csv.writer(predictions_file)
    open_file_object.writerow(["Id","Prediction"])
    open_file_object.writerows(zip(test_ids, predictions))
    predictions_file.close()

    print "Prediction file written."