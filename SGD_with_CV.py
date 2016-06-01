import numpy as np
import csv
from datetime import datetime
import matplotlib.pyplot as plt
from numpy.linalg import matrix_rank
from sklearn.cluster import KMeans
import sys
import random
import os.path

BEST_K = 15
NMB_OF_TRAINING_ITERATIONS = 10000000
REG_LINSPACE_SIZE = 3
K_LINSPACE_SIZE = 5

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

    #returned are u_d of size 1*k 
    #z_n of size k*1
    return u_d,z_n


if (not os.path.isfile("training_ids.npy") or not os.path.isfile("validation_ids.npy")):
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

else:
    training_ids = np.load("training_ids.npy")
    validation_ids = np.load("validation_ids.npy")
    print "Data loaded from files"

#do a CV
rand_ids = np.random.choice(range(0,len(training_ids)), size=NMB_OF_TRAINING_ITERATIONS)

reg_terms = np.linspace(0, 1, num=REG_LINSPACE_SIZE)
kSpace = np.linspace(3, 40, num=K_LINSPACE_SIZE).astype(int)
print kSpace

mses = np.zeros((len(kSpace),len(reg_terms)))
for k in range(0,len(kSpace)):
    for i in range(0,len(reg_terms)):
        reg_term = reg_terms[i]

        #numbers grow to big!!! -> nan in the solution!
        U = np.random.rand(1000,kSpace[k])
        Z = np.random.rand(10000,kSpace[k])

        j = 1 #initialization to zero will result in devide by zero 
        for rand_idx in rand_ids:
            training_id = training_ids[rand_idx]
            nz_item = training_id[0]
            d = nz_item[0]
            n = nz_item[1]
            rating = nz_item[2]
            
            U[d,:],Z[n,:] = sgd(rating,U[d,:],Z[n,:],0.001, reg_term)

            if (np.mod(j,500000) == 0):
                print j
            j = j + 1

        print "Optimization done."


        #use the truncated prediction matrix (obtained by the best k singular values observed from the plot)
        prediction_matrix = np.dot(U,Z.T)

        print prediction_matrix
        print np.min(prediction_matrix), np.max(prediction_matrix), np.mean(prediction_matrix)
        print len(prediction_matrix[np.where(prediction_matrix < 0)])
        print len(prediction_matrix[np.where(prediction_matrix > 5)])

        prediction_matrix[np.where(prediction_matrix < 0)] = 0
        prediction_matrix[np.where(prediction_matrix > 5)] = 5
        mses[k,i] = irmse(prediction_matrix,validation_ids)
        print "The overall RMSE prediction error for selected " + str(kSpace[k]) + " optimized singular values and reg_term " + str(reg_term) +" is: " + str(mses[k,i])


np.save("mses", mses)
for k in range(0,len(kSpace)):
    plt.plot(reg_terms, mses[k,:], color=(col,col,col))
    plt.show()