#import libraries
import numpy as np
import csv
from datetime import datetime
import matplotlib.pyplot as plt
import sys
import random
import matplotlib
from scipy.spatial import KDTree
from sklearn import preprocessing

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

EPS:
For early abort. It training error falls under the value of eps.

KNN_ITEM and KNN_USER:
Set number of KNNs to be searched.

VALIDATION:
Set to True in order to validate our model. Set to False in order to use the whole dataset for training.

USE_KNN_ITEM_USER:
Set to True in order to obtain rating mixed from user and item neighbourhood information.

NUM_ALPHAS:
Set to number of alphas to search. 
Used in alpha*item_rating_neighbourhood + (1-alpha)*user_rating_neighbourhood. Grid search for optimal alpha in alphas. EXPENSIVE!

OPTIMAL_ALPHA:
Set optimal alpha to the value observed while training on the training set and validating on the validation set. Set it to retrain the model
on the whole data set.

VALIDATION_SET_SIZE:
Set to number between 0 and 1. Defines percentage of the dataset used for validating the model.

ROUND:
If True round to next nearest integer if sure that it should be the next one. E.g 3.05 rounded to 3; 3.95 rounded to 4; but everything in between not
sure so it stays a float rating.

NORMALIZE_ITEM_USER:
Set to True to normalize the user and item vectors in the low dimensional space to unit length.

SET_EARLY_ABORT_RAISE:
Set it to an integer of number of continously increasing validation scores allowed before aborting.

USE_IMPROVED_SGD:
Set it to True in roder to account for biases in the rating calculation: According to the Simon's Funk algorithm.
"""

#constants to be adapted
SUBMISSION_FILENAME = "inc_svd_knn_user_item_differentKNN_directPrediction_submission.csv"
GENERATE_SUBMISSION = True

#optimal parameters
BEST_K = 5
KNN_ITEM = 999
KNN_USER = 9999
LEARNING_RATE = 0.005                                                                                                                                                                                                                                                                                            
NMB_OF_TRAINING_ITERATIONS = 50000000
SEED_NUM = 500
REGULARIZATION_TERM = 0
EPS = 0.1
OPTIMAL_ALPHA = 0.5
NUM_ALPHAS = 5
ROUND = True
NORMALIZE_ITEM_USER = False
USE_IMPROVED_SGD = True

#training
GRID_SEARCH = True
K = 50
REG_TERMS = 3
VALIDATION = True
USE_KNN_ITEM_USER = True
VALIDATION_SET_SIZE = 0.2
SET_EARLY_ABORT_RAISE = 4

#do not modify parameters
alphas = np.linspace(0,1,NUM_ALPHAS) 

#round adapted to the problem. Round to next nearest integer if sure that it should be the next one. E.g 3.05 rounded to 3; 3.95 rounded to 4; but everything in between not
#sure so it stays a float rating
def iround(ratings,ROUND,is_scalar):
    if(ROUND and (not is_scalar)):
        zero_round = np.where(ratings.astype(int) == 0)
        ratings[zero_round] = 1
        indices_round_down = np.where((ratings%ratings.astype(int))<= 0.05)
        indices_round_up = np.where((ratings%ratings.astype(int))>= 0.95)
        ratings[indices_round_down] = ratings[indices_round_down].astype(int)
        ratings[indices_round_up] = ratings[indices_round_up].astype(int) + 1
    elif(ROUND and is_scalar):
        if(int(ratings) == 0):
            ratings = 1
            return ratings
        if(ratings%int(ratings) <= 0.05):
            ratings = int(ratings)
        elif(ratings%int(ratings) >= 0.95):
            ratings= int(ratings) + 1

    return ratings

#used for sorting
def get_key(item):
    return item[1]
    
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

#calculate the sgd 
def sgd(x_dn,u_d,z_n,stepsize,reg_term):
    dot_prod = np.dot(u_d,z_n)

    grad_u_d = -1*(x_dn-dot_prod)*z_n + 2*reg_term*u_d
    if (np.any(np.isnan(grad_u_d))):
        print x_dn, u_d, z_n
        sys.exit()

    grad_z_n = -1*(x_dn-dot_prod)*u_d + 2*reg_term*z_n

    u_d = u_d - stepsize*grad_u_d
    z_n = z_n - stepsize*grad_z_n

    return u_d,z_n

def improved_sgd(x_dn,u_d,z_n,b_u,b_i,mu,stepsize,reg_term):
    dot_prod =  np.dot(u_d,z_n)
    p_dn = x_dn-(mu+b_i+b_u+dot_prod)

    grad_u_d = -1*(p_dn)*z_n + 2*reg_term*u_d
    if (np.any(np.isnan(grad_u_d))):
        print x_dn, u_d, z_n
        sys.exit()

    grad_z_n = -1*(p_dn)*u_d + 2*reg_term*z_n
    grad_b_u = -1*p_dn+2*reg_term*b_u
    grad_b_i = -1*p_dn+2*reg_term*b_i
    grad_mu = -1*p_dn+2*reg_term*mu

    u_d = u_d - stepsize*grad_u_d
    z_n = z_n - stepsize*grad_z_n
    b_u = b_u - stepsize*grad_b_u
    b_i = b_i - stepsize*grad_b_i
    mu = mu - stepsize*grad_mu

    return u_d,z_n,b_u,b_i,mu

#set the seed to get determinism
np.random.seed(SEED_NUM)

#load data
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
    if(rand_num < VALIDATION_SET_SIZE and VALIDATION):
       validation_ids.append([(i,j,int(train_predictions[k]))])
    else:
       training_ids.append([(i,j,int(train_predictions[k]))])
       train_matrix[i,j] = int(train_predictions[k])
       
print "Data matrix built."
if(VALIDATION):
    print "Number of items in the validation set: " + str(len(validation_ids)) + "."
print "Number of items in the training set: " + str(len(training_ids)) + "."

#prepare random draws from data set
rand_ids = np.random.choice(range(0,len(training_ids)), size=NMB_OF_TRAINING_ITERATIONS)
print "Random ids collected"


#grid search for optimal hyperparameteres k and reg. term using sgd algorithm. Uses validation set.
if(GRID_SEARCH):
    VALIDATION = True

if(GRID_SEARCH and VALIDATION):
    k_search = np.linspace(1,K,K).astype(int)
    reg_terms = np.linspace(0,0.05,REG_TERMS)
    rmse = np.zeros((K,REG_TERMS))
    for reg_term in range(len(reg_terms)):
        for k in range(len(k_search)):
            U = np.random.rand(1000,k_search[k])
            Z = np.random.rand(10000,k_search[k])
            if(USE_IMPROVED_SGD):
                b_u = np.random.rand(1,10000)
                b_i = np.random.rand(1000,1)
                mu = np.random.rand(1)

            j = 1
            validate_err_curr = np.inf
            validate_err_prev = np.inf
            training_err_curr = np.inf
            training_err_prev = np.inf
            count_raise = 0
            for rand_idx in xrange(len(rand_ids)):
                training_id = training_ids[rand_ids[rand_idx]]
                nz_item = training_id[0]
                d = nz_item[0]
                n = nz_item[1]
                rating = nz_item[2]
                
                if(USE_IMPROVED_SGD):
                    U[d,:],Z[n,:],b_u[0,n],b_i[d,0],mu = improved_sgd(rating,U[d,:],Z[n,:],b_u[0,n],b_i[d,0],mu,LEARNING_RATE, reg_terms[reg_term])
                else:
                    U[d,:],Z[n,:] = sgd(rating,U[d,:],Z[n,:],LEARNING_RATE, reg_terms[reg_term])


                if (np.mod(j,500000) == 0 or j == 1):
                    if(USE_IMPROVED_SGD):
                        prediction_matrix = np.dot(U,Z.T) + np.tile(b_u,(1000,1)) + np.tile(b_i,(1,10000)) + np.tile(mu,(1000,10000))

                    else:
                        prediction_matrix = np.dot(U,Z.T)

                    prediction_matrix = iround(prediction_matrix,ROUND,False)
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

                        if(count_raise < SET_EARLY_ABORT_RAISE):
                            count_raise = count_raise + 1
                        else:
                            break
                    else:
                        print "At iteration: " + str(j) + "."
                        count_raise = 0

                j = j + 1
            
            if(USE_IMPROVED_SGD):
                prediction_matrix = np.dot(U,Z.T) + np.tile(b_u,(1000,1)) + np.tile(b_i,(1,10000)) + np.tile(mu,(1000,10000))

            else:
                prediction_matrix = np.dot(U,Z.T)

            prediction_matrix = iround(prediction_matrix,ROUND,False)
            rmse[k,reg_term] = irmse(prediction_matrix,validation_ids)
        print "Done with regularization " + str((float(reg_term)/float(len(reg_terms)))*100) + "%"


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


#read best REGULARIZATION_TERM and K from the obtained graph. Using this optimal incremental SVD setting continue using the adapted KNN method.


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
#generate submission with optimal configuration and using tonly the training set for training (80% of the given dataset)
if(GENERATE_SUBMISSION and VALIDATION):
    #run the optimization
    U = np.random.rand(1000,BEST_K)
    Z = np.random.rand(10000,BEST_K)
    if(USE_IMPROVED_SGD):
        b_u = np.random.rand(1,10000)
        b_i = np.random.rand(1000,1)
        mu = np.random.rand(1)
    j = 1
    validate_err_curr = np.inf
    validate_err_prev = np.inf
    training_err_curr = np.inf
    training_err_prev = np.inf
    count_raise = 0
    for rand_idx in xrange(len(rand_ids)):
        training_id = training_ids[rand_ids[rand_idx]]
        nz_item = training_id[0]
        d = nz_item[0]
        n = nz_item[1]
        rating = nz_item[2]
        
        if(USE_IMPROVED_SGD):
            U[d,:],Z[n,:],b_u[0,n],b_i[d,0],mu = improved_sgd(rating,U[d,:],Z[n,:],b_u[0,n],b_i[d,0],mu,LEARNING_RATE, REGULARIZATION_TERM)
        else:
            U[d,:],Z[n,:] = sgd(rating,U[d,:],Z[n,:],LEARNING_RATE, REGULARIZATION_TERM)

        if (np.mod(j,500000) == 0 or j == 1):
            if(USE_IMPROVED_SGD):
                prediction_matrix = np.dot(U,Z.T) + np.tile(b_u,(1000,1)) + np.tile(b_i,(1,10000)) + np.tile(mu,(1000,10000))

            else:
                prediction_matrix = np.dot(U,Z.T)

            prediction_matrix = iround(prediction_matrix,ROUND,False)
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
                
                if(count_raise < SET_EARLY_ABORT_RAISE):
                    count_raise = count_raise + 1
                else:
                    break
            else:
                print "At iteration: " + str(j) + "."
                print "Current validation error: " + str(validate_err_curr) + "."
                print "Previous validation error: " + str(validate_err_prev) + "."

                print "Current training error: " + str(training_err_curr) + "."
                print "Previous training error: " + str(training_err_prev) + "."
                count_raise = 0


        j = j + 1
    if(USE_IMPROVED_SGD):
        prediction_matrix = np.dot(U,Z.T) + np.tile(b_u,(1000,1)) + np.tile(b_i,(1,10000)) + np.tile(mu,(1000,10000))

    else:
        prediction_matrix = np.dot(U,Z.T)
    print "Optimization done."
    
    #NN(#data x #dim) algorithm on the user matrix Z. For a given user, find optimal n NN users and use weighted (normalized distances) 
    #average of their existing ratings to obtain the rating for the observed user; perform the same for the items. Try to find
    # a linear combination between those two obtained ratings.if both nn are not found then use the rating from the prediction matrix 
    #with added bias accordingly(multiplication of U and Z with an additive correction term)
    
    predictions = []
    points_index_user = [[None for x in range(KNN_USER)] for y in range(10000)]
    points_index_item = [[None for x in range(KNN_ITEM)] for y in range(1000)]
    dist_user = [[None for x in range(KNN_USER)] for y in range(10000)]
    dist_item = [[None for x in range(KNN_ITEM)] for y in range(1000)]
    if(NORMALIZE_ITEM_USER):
        U = preprocessing.normalize(U, norm='l2')
        Z = preprocessing.normalize(Z, norm='l2')
    tree_user = KDTree(Z)
    tree_item = KDTree(U)

    if(USE_KNN_ITEM_USER):
        validation_error = []
        num_of_items = len(validation_ids)
        for alpha in alphas:
            j = 1
            error = 0
            for validation_id in validation_ids:
                item_data = validation_id[0]
                item = item_data[0]
                user = item_data[1]
                item_rating = item_data[2]

                if (points_index_user[user][0] == None):
                    dist_user[user], points_index_user[user] = tree_user.query(Z[user,:],k=KNN_USER,p=2)
                if (points_index_item[item][0] == None):
                    dist_item[item], points_index_item[item] = tree_item.query(U[item,:],k=KNN_ITEM,p=2)
                norm_factor_user = 0
                norm_factor_item = 0
                knn_possible_user = False
                knn_possible_item = False
                for i in range(KNN_USER):
                    nn_user = points_index_user[user][i]
                    if(train_matrix[item,nn_user] != 0):
                        norm_factor_user = norm_factor_user + 1./dist_user[user][i]
                        knn_possible_user = True
                      
                for i in range(KNN_ITEM):
                    nn_item = points_index_item[item][i]
                    if(train_matrix[nn_item,user] != 0):
                        norm_factor_item = norm_factor_item + 1./dist_item[item][i]
                        knn_possible_item = True
                    
                if(knn_possible_user and knn_possible_item):
                    rating_user = 0
                    rating_item = 0
                    for i in range(KNN_USER):
                        nn_user = points_index_user[user][i]
                        if(train_matrix[item,nn_user] != 0):
                            dist_norm = 1./dist_user[user][i]
                            weight = float(dist_norm)/float(norm_factor_user)
                            rating_user = rating_user + weight*train_matrix[item,nn_user]
                        

                    for i in range(KNN_ITEM):
                        nn_item = points_index_item[item][i]
                        if(train_matrix[nn_item,user] != 0):
                            dist_norm = 1./dist_item[item][i]
                            weight = float(dist_norm)/float(norm_factor_item)
                            rating_item = rating_item + weight*train_matrix[nn_item,user]

                    #linear combination of both ratings of items and users
                    rating = rating_item*alpha + rating_user*(1-alpha)

                elif(knn_possible_user):
                    rating = 0
                    for i in range(KNN_USER):
                        nn_user = points_index_user[user][i]
                        if(train_matrix[item,nn_user] != 0):
                            dist_norm = 1./dist_user[user][i]
                            weight = float(dist_norm)/float(norm_factor_user)
                            rating = rating + weight*train_matrix[item,nn_user]
                
                elif(knn_possible_item):
                    rating = 0
                    for i in range(KNN_ITEM):
                        nn_item = points_index_item[item][i]
                        if(train_matrix[nn_item,user] != 0):
                            dist_norm = 1./dist_item[item][i]
                            weight = float(dist_norm)/float(norm_factor_item)
                            rating = rating + weight*train_matrix[nn_item,user]
                
                else:
                    rating = prediction_matrix[item,user]
                    if(rating > 5):
                        rating = 5
                    elif(rating < 1):
                        rating = 1
            
                if(j % 100000 == 0):
                    print "Predictions computing..." + str(float(j)/float(len(validation_ids)))
                
                rating = iround(rating,ROUND,True)
                error = error + (np.square(rating-item_rating))
                j = j + 1
            
            validation_error.append(np.sqrt(float(error)/float(num_of_items)))
            print "Validation error for alpha " + str(alpha) + " is: " + str(np.sqrt(float(error)/float(num_of_items)))
        
        best_alpha = alphas[validation_error.index(min(validation_error))]
        print "best alpha is:", best_alpha
        #prediction
        #user KNN
        #only consider user once (users sorted), and store his neighbours to save computation time
        j = 1
        for prediction_index in prediction_indices:
            item = prediction_index[0]
            user = prediction_index[1]
            if (points_index_user[user][0] == None):
                dist_user[user], points_index_user[user] = tree_user.query(Z[user,:],k=KNN_USER,p=2)
            if (points_index_item[item][0] == None):
                dist_item[item], points_index_item[item] = tree_item.query(U[item,:],k=KNN_ITEM,p=2)
                
            norm_factor_user = 0
            norm_factor_item = 0
            knn_possible_user = False
            knn_possible_item = False
            for i in range(KNN_USER):
                nn_user = points_index_user[user][i]
                if(train_matrix[item,nn_user] != 0):
                    norm_factor_user = norm_factor_user + 1./dist_user[user][i]
                    knn_possible_user = True
                  
            for i in range(KNN_ITEM):
                nn_item = points_index_item[item][i]
                if(train_matrix[nn_item,user] != 0):
                    norm_factor_item = norm_factor_item + 1./dist_item[item][i]
                    knn_possible_item = True
            
            if(knn_possible_user and knn_possible_item):
                rating_user = 0
                rating_item = 0
                for i in range(KNN_USER):
                    nn_user = points_index_user[user][i]
                    if(train_matrix[item,nn_user] != 0):
                        dist_norm = 1./dist_user[user][i]
                        weight = float(dist_norm)/float(norm_factor_user)
                        rating_user = rating_user + weight*train_matrix[item,nn_user]
                    

                for i in range(KNN_ITEM):
                    nn_item = points_index_item[item][i]
                    if(train_matrix[nn_item,user] != 0):
                        dist_norm = 1./dist_item[item][i]
                        weight = float(dist_norm)/float(norm_factor_item)
                        rating_item = rating_item + weight*train_matrix[nn_item,user]
                #linear combination of both ratings of items and users
                rating = rating_item*best_alpha + rating_user*(1-best_alpha)
                predictions.append(rating)
            
            elif(knn_possible_user):
                rating = 0
                for i in range(KNN_USER):
                    nn_user = points_index_user[user][i]
                    if(train_matrix[item,nn_user] != 0):
                        dist_norm = 1./dist_user[user][i]
                        weight = float(dist_norm)/float(norm_factor_user)
                        rating = rating + weight*train_matrix[item,nn_user]
                predictions.append(rating)
            
            elif(knn_possible_item):
                rating = 0
                for i in range(KNN_ITEM):
                    nn_item = points_index_item[item][i]
                    if(train_matrix[nn_item,user] != 0):
                        dist_norm = 1./dist_item[item][i]
                        weight = float(dist_norm)/float(norm_factor_item)
                        rating = rating + weight*train_matrix[nn_item,user]
                predictions.append(rating)
                
            else:
                rating = prediction_matrix[item,user]
                if(rating > 5):
                    rating = 5
                elif(rating < 1):
                    rating = 1
                predictions.append(rating)
            
            if(j % 100000 == 0):
                print "Predictions computing..." + str(float(j)/float(len(prediction_indices)))
        
            j = j + 1
    else:
        for prediction_index in prediction_indices:
            rating = prediction_matrix[prediction_index[0],prediction_index[1]]
            if(rating > 5):
                rating = 5
            elif(rating < 1):
                rating = 1
            
            predictions.append(rating)
    
#generate submission with optimal configuration and using the whole dataset provided
if(GENERATE_SUBMISSION and (not VALIDATION)):
    #run the optimization
    U = np.random.rand(1000,BEST_K)
    Z = np.random.rand(10000,BEST_K)
    if(USE_IMPROVED_SGD):
        b_u = np.random.rand(1,10000)
        b_i = np.random.rand(1000,1)
        mu = np.random.rand(1)

    j = 1
    training_err_curr = np.inf
    training_err_prev = np.inf
    count_raise = 0
    for rand_idx in xrange(len(rand_ids)):
        training_id = training_ids[rand_ids[rand_idx]]
        nz_item = training_id[0]
        d = nz_item[0]
        n = nz_item[1]
        rating = nz_item[2]
        
        if(USE_IMPROVED_SGD):
            U[d,:],Z[n,:],b_u[0,n],b_i[d,0],mu = improved_sgd(rating,U[d,:],Z[n,:],b_u[0,n],b_i[d,0],mu,LEARNING_RATE, REGULARIZATION_TERM)
        else:
            U[d,:],Z[n,:] = sgd(rating,U[d,:],Z[n,:],LEARNING_RATE, REGULARIZATION_TERM)


        if (np.mod(j,500000) == 0 or j == 1):
            if(USE_IMPROVED_SGD):
               prediction_matrix = np.dot(U,Z.T) + np.tile(b_u,(1000,1)) + np.tile(b_i,(1,10000)) + np.tile(mu,(1000,10000))

            else:
                prediction_matrix = np.dot(U,Z.T)

            prediction_matrix = iround(prediction_matrix,ROUND,False)
            training_err_prev = training_err_curr
            training_err_curr = irmse(prediction_matrix,training_ids)
            if(training_err_prev < training_err_curr or training_err_curr < EPS):
                print "Early abort."
               
                print "Current training error: " + str(training_err_curr) + "."
                print "Previous training error: " + str(training_err_prev) + "."
               
                if(count_raise < SET_EARLY_ABORT_RAISE):
                    count_raise = count_raise + 1
                else:
                    break
            else:
                print "At iteration: " + str(j) + "."
                count_raise = 0

        j = j + 1
    if(USE_IMPROVED_SGD):
        prediction_matrix = np.dot(U,Z.T) + np.tile(b_u,(1000,1)) + np.tile(b_i,(1,10000)) + np.tile(mu,(1000,10000))

    else:
        prediction_matrix = np.dot(U,Z.T)

    print "Optimization done."

    predictions = []
    if(USE_KNN_ITEM_USER):
        #prediction
        #only consider user once (users sorted), and store his neighbours to save computation time

        points_index_user = [[None for x in range(KNN_USER)] for y in range(10000)]
        points_index_item = [[None for x in range(KNN_ITEM)] for y in range(1000)]
        dist_user = [[None for x in range(KNN_USER)] for y in range(10000)]
        dist_item = [[None for x in range(KNN_ITEM)] for y in range(1000)]
        if(NORMALIZE_ITEM_USER):
            U = preprocessing.normalize(U, norm='l2')
            Z = preprocessing.normalize(Z, norm='l2')
        tree_user = KDTree(Z)
        tree_item = KDTree(U)

        j = 1
        for prediction_index in prediction_indices:
            item = prediction_index[0]
            user = prediction_index[1]
            if (points_index_user[user][0] == None):
                dist_user[user], points_index_user[user] = tree_user.query(Z[user,:],k=KNN_USER,p=2)
            if (points_index_item[item][0] == None):
                dist_item[item], points_index_item[item] = tree_item.query(U[item,:],k=KNN_ITEM,p=2)
                
            norm_factor_user = 0
            norm_factor_item = 0
            knn_possible_user = False
            knn_possible_item = False
            for i in range(KNN_USER):
                nn_user = points_index_user[user][i]
                if(train_matrix[item,nn_user] != 0):
                    norm_factor_user = norm_factor_user + 1./dist_user[user][i]
                    knn_possible_user = True
                  
            for i in range(KNN_ITEM):
                nn_item = points_index_item[item][i]
                if(train_matrix[nn_item,user] != 0):
                    norm_factor_item = norm_factor_item + 1./dist_item[item][i]
                    knn_possible_item = True
            
            if(knn_possible_user and knn_possible_item):
                rating_user = 0
                rating_item = 0
                for i in range(KNN_USER):
                    nn_user = points_index_user[user][i]
                    if(train_matrix[item,nn_user] != 0):
                        dist_norm = 1./dist_user[user][i]
                        weight = float(dist_norm)/float(norm_factor_user)
                        rating_user = rating_user + weight*train_matrix[item,nn_user]
                    

                for i in range(KNN_ITEM):
                    nn_item = points_index_item[item][i]
                    if(train_matrix[nn_item,user] != 0):
                        dist_norm = 1./dist_item[item][i]
                        weight = float(dist_norm)/float(norm_factor_item)
                        rating_item = rating_item + weight*train_matrix[nn_item,user]
                #linear combination of both ratings of items and users
                rating = rating_item*OPTIMAL_ALPHA + rating_user*(1-OPTIMAL_ALPHA)
                predictions.append(rating)
            
            elif(knn_possible_user):
                rating = 0
                for i in range(KNN_USER):
                    nn_user = points_index_user[user][i]
                    if(train_matrix[item,nn_user] != 0):
                        dist_norm = 1./dist_user[user][i]
                        weight = float(dist_norm)/float(norm_factor_user)
                        rating = rating + weight*train_matrix[item,nn_user]
                predictions.append(rating)
            
            elif(knn_possible_item):
                rating = 0
                for i in range(KNN_ITEM):
                    nn_item = points_index_item[item][i]
                    if(train_matrix[nn_item,user] != 0):
                        dist_norm = 1./dist_item[item][i]
                        weight = float(dist_norm)/float(norm_factor_item)
                        rating = rating + weight*train_matrix[nn_item,user]
                predictions.append(rating)
                
            else:
                rating = prediction_matrix[item,user]
                if(rating > 5):
                    rating = 5
                elif(rating < 1):
                    rating = 1
                predictions.append(rating)
            
            if(j % 100000 == 0):
                print "Predictions computing..." + str(float(j)/float(len(prediction_indices)))
        
            j = j + 1

    else:
        for prediction_index in prediction_indices:
            rating = prediction_matrix[prediction_index[0],prediction_index[1]]
            if(rating > 5):
                rating = 5
            elif(rating < 1):
                rating = 1
            
            predictions.append(rating)
    
    
print "Predictions ready."

#generate prediction file
predictions = np.array(predictions)
predictions = iround(predictions,ROUND,False)
predictions_file = open("submissions/" + SUBMISSION_FILENAME, "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["Id","Prediction"])
open_file_object.writerows(zip(test_ids, predictions))
predictions_file.close()

print "Prediction file written."
