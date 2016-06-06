import numpy as np
import csv
import matplotlib
from datetime import datetime
import matplotlib.pyplot as plt
from numpy.linalg import matrix_rank
from sklearn.cluster import KMeans
import sys
import random
import os.path
from math import sqrt,pi,exp

BEST_K = 5
NMB_OF_TRAINING_ITERATIONS = 10000
LEARNING_RATE = 0.001
SIGMA = 0.01
SUBMISSION_FILENAME = "test.csv"

np.random.seed(500)

def getKey(item):
    return item[0]

def getUserDistance(userA, usersB):
    normUserA = np.linalg.norm(userA, ord=2)
    normsUsersB = np.linalg.norm(usersB, ord=2, axis=1)
    unnormalized = np.divide(np.dot(usersB, userA), normUserA)

    return 1 - np.abs(np.divide(unnormalized, normsUsersB));
    #return np.linalg.norm(userA-userB, ord=2)

def dnorm(X,mu=0,sigma=1.5):
    """
    Named after the dnorm function from programming language R.
    Density generation for the normal distribution with mean = mu and standard deviation = sigma.
    Input: 
     - *x* : vector or integer of 'x-axis' values
     - *mu* [default = 0]
     - *sigma* [defaul = 1.5]
    Output: 
     - float or list with floats containing the calculated values
    """
    dnorm_list = []  
    constant = 1.0/sqrt(2.0*pi*sigma**2)      # Calc once
    denominator = (2.0*sigma**2)          # Calc once
    if type(X) == list:               # For a list of int/floats
        for x in X:
            value = constant*exp(-((x-mu)**2)/denominator)
            dnorm_list.append(value) 
        return dnorm_list    
    elif type(X) == int or float:         # For a single int/float
        value = constant*exp(-((X-mu)**2)/denominator)
        return value

    else:
        print "The first argument (X) must be a list with integers/floats or a single integer/float.\n"
        sys.exit()    

def normalized_dnorm(X,mu=0,sigma=1.5):
    """
    Generate normalized values based on the maximum of the bell-shaped normal distribution. 
    It uses the self defined dnorm function to calculate these values.
    Input: 
     - *x* : vector or integer of 'x-axis' values
     - *mu* [default = 0]
     - *sigma* [defaul = 1.5]*
    Output: 
     - float or list with floats containing the calculated values
    """
    unnormalized = dnorm(X,mu,sigma)      # Do unnormalized dnorm
    max_ = 1.0/sqrt(2.0*pi*sigma**2)      # Max value at x = 0

    norm_dnorm_list = []
    if type(X) == list:               # For a list of int/floats
        for x in unnormalized:
            norm_dnorm_list.append(x/max_)
        
    elif type(X) == int or float:         # For a single int/float
        return unnormalized/max_

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

def sgdBatch(x_dn,u_d,z_n,stepsize, reg_term):
    batchSize = len(x_dn)
    
    for i in range(0,batchSize):
        dotProd = np.dot(u_d[i],z_n[i])

        grad_u_d += -(x_dn[i]-dotProd)*z_n[i] + 2*reg_term*u_d[i]

        grad_z_n += -(x_dn[i]-dotProd)*u_d[i] + 2*reg_term*z_n[i]

    grad_u_d /= batchSize
    grad_z_n /= batchSize

    u_d = u_d - stepsize*grad_u_d
    z_n = z_n - stepsize*grad_z_n

    return u_d,z_n


if (not os.path.isfile("training_ids.npy") or not os.path.isfile("validation_ids.npy") or not os.path.isfile("test_ids.npy")):
    #data loading
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
    train_matrix_keep = np.zeros((1000,10000))

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
            train_matrix_keep[i,j] = int(train_predictions[k])

    print "Data matrix built."
    print "Number of items in the validation set: " + str(len(validation_ids))
    print "Number of items in the training set: " + str(np.count_nonzero(train_matrix))

    np.save("training_ids", training_ids)
    np.save("validation_ids", validation_ids)
    np.save("test_ids", test_ids)

else:
    training_ids = np.load("training_ids.npy")
    validation_ids = np.load("validation_ids.npy")
    test_ids = np.load("test_ids.npy")
    print "Data loaded from files"
    #plt.hist(training_ids[:,0,1], 10000)
    #plt.show()


rand_ids = np.random.choice(range(0,len(training_ids)), size=NMB_OF_TRAINING_ITERATIONS)
 
U = np.random.rand(1000,BEST_K)
Z = np.random.rand(10000,BEST_K)

j = 1 #initialization to zero will result in devide by zero 
for rand_idx in rand_ids:
    training_id = training_ids[rand_idx]
    nz_item = training_id[0]
    d = nz_item[0]
    n = nz_item[1]
    rating = nz_item[2]
    
    U[d,:],Z[n,:] = sgd(rating,U[d,:],Z[n,:],LEARNING_RATE, 0)

    if (np.mod(j,500000) == 0):
        print j
    j = j + 1

print "Optimization done."


#use the truncated prediction matrix (obtained by the best k singular values observed from the plot)
prediction_matrix = np.dot(U,Z.T)

#print prediction_matrix
print np.min(prediction_matrix), np.max(prediction_matrix), np.mean(prediction_matrix)
print len(prediction_matrix[np.where(prediction_matrix < 1)])
print len(prediction_matrix[np.where(prediction_matrix > 5)])

prediction_matrix[np.where(prediction_matrix < 1)] = 1
prediction_matrix[np.where(prediction_matrix > 5)] = 5
mses = irmse(prediction_matrix,validation_ids)
print "The overall RMSE prediction error for selected " + str(BEST_K) + " optimized singular values and reg_term " + str(0) +" is: " + str(mses)


#compute validation score
#how?!?

#find the indices of the items we have to predict and store them
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


############################
#do NN stuff
############################
#find the predictions
counter = 1
prevMovieIdx = -1
predictions = []
for prediction_index in prediction_indices:
    movieIdx = prediction_index[0]
    userIdx = prediction_index[1]

    currentUser = Z[userIdx,:]

    if (movieIdx != prevMovieIdx):
        compare_ids = np.where(training_ids[:,0,0] == movieIdx)[0]
        idx = training_ids[compare_ids,0,1]
        compareUsers = Z[idx,:]

    d = getUserDistance(currentUser, compareUsers)
    weights = dnorm(d.tolist(), mu=0, sigma=SIGMA)

    #normalize weights
    weights = weights / np.sum(weights)

    
    ratings = training_ids[compare_ids,0,2]
    final_rating = np.dot(weights,ratings)

    predictions.append(min(5,max(1,final_rating)))

    prevMovieIdx = movieIdx

    counter += 1
    if (np.mod(counter, 1000) == 0):
        print counter
        print final_rating        



#generate prediction file
predictions_file = open("submissions/" + SUBMISSION_FILENAME, "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["Id","Prediction"])
open_file_object.writerows(zip(test_ids, predictions))
predictions_file.close()
print "Prediction file written."