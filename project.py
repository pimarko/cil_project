import numpy as np
import csv
from datetime import datetime
import matplotlib.pyplot as plt
from numpy.linalg import matrix_rank
from sklearn.cluster import KMeans
import sys
import random

#general comments
# - ratings as floats returned (maybe try ints)!
# - maybe try different initialization technique
# - pick different BEST_K observed from the plot
# - try different APPROACH_SOLUTION
# - pick finer alphas in the mixed initilization 
# - try to imput missing values with user,item mix and then apply svd svd and then GD to imporve U,V!

#constants
GENERATE_SUBMISSION = True

GENERATE_PLOT = False

BASELINE_SOLUTION = False
SVD_TRUNCATION_SOLUTION = False
SVD_IMPROVED_SOLUTION = True
K_MEANS_SOLUTION = False

eps = 0.1
BEST_K = 8 #for svd and k_means
K = 30 #for k_means and svd
EPOCHE = 50
INITILIZATION_ITEM_AVG = False
INITILIZATION_USER_AVG = False
INITILIZATION_ITEM_USER_MIX = False
INITILIZATION_ITEM_USER_AVG = False
now = datetime.now()

if(BASELINE_SOLUTION):
	SUBMISSION_FILENAME = "baseline_solution_submission_" + str(now.day) +"_" + str(now.hour) + \
	"_" + str(now.minute) + "_" + str(now.second) +".csv"
elif(SVD_TRUNCATION_SOLUTION):
	SUBMISSION_FILENAME = "svd_truncation_solution_submission_" + str(now.day) +"_" + str(now.hour) + \
	"_" + str(now.minute) + "_" + str(now.second) +".csv"
elif(K_MEANS_SOLUTION):
	SUBMISSION_FILENAME = "k_means_solution_submission_" + str(now.day) +"_" + str(now.hour) + \
	"_" + str(now.minute) + "_" + str(now.second) +".csv"
elif(SVD_IMPROVED_SOLUTION):
	SUBMISSION_FILENAME = "svd_improved_solution_submission_" + str(now.day) +"_" + str(now.hour) + \
	"_" + str(now.minute) + "_" + str(now.second) +".csv"
else:
	sys.exit("For now no other method for rating prediction exists. Select the BASELINE_SOLUTION or SVD_TRUNCATION_SOLUTION")

#definitions
def compute_missing_rating(ratings):
	ratings = np.array(ratings)
	s = float(np.sum(ratings))
	l = float(len(np.where(ratings != 0)[0]))

	if(l == 0):
		return 0.0
	else:
		return s/l

def get_square_root(elems):
	elems_square_root = []
	for elem in elems:
		elems_square_root.append(float(np.sqrt(elem)))

	elems_square_root = np.array(elems_square_root)	
	return elems_square_root

def iround(x):
	return int(round(x) - .5) + (x > 0)

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

def is_in_cluster(user_index,labels):
	users_in_cluster = []
	user_cluster = labels[user_index]
	for i in range(len(labels)):
		if(i != user_index and user_cluster == labels[i]):
			users_in_cluster.append(i)

	return users_in_cluster

def get_cluster_item_avg(train_matrix_keep,labels,validation_ids):
	tmp_train_matrix = np.zeros((1000,10000))
	tmp_train_matrix = train_matrix_keep
	labels = np.array(labels)
	zero_row_indices = np.where(train_matrix_keep == 0)[0]
	zero_column_indices = np.where(train_matrix_keep == 0)[1]																																																																																																																																																																																																																																																																		

	zero_item_indices = zip(zero_row_indices,zero_column_indices)

	for validation_id in validation_ids:
		item = validation_id[0]
		positions = np.where(labels == labels[item[1]])
		train_matrix_keep[item[0],item[1]] = compute_missing_rating(tmp_train_matrix[item[0],positions])

	return train_matrix_keep

def optimize(U,V,X):
	U_sym = np.dot(U.T,U)

	U_x = np.dot(U.T,X)
	V_sym = np.dot(V,V.T)
	V_x = np.dot(V,X.T)
	V = np.linalg.solve(U_sym,U_x)
	U = np.linalg.solve(V_sym,V_x)

	return U,V

def sgd(x_dn,u_d,z_n,stepsize):
	#stepsize to be set
	#z_n size is k*1
	#u_d size is 1*k

	grad_u_d = np.dot(x_dn - np.dot(u_d,z_n),z_n.T)

	grad_z_n = np.dot((x_dn - np.dot(u_d,z_n)),u_d.T)

	u_d = u_d - stepsize*grad_u_d
	z_n = z_n - stepsize*grad_z_n

	#returned are u_d of size 1*k 
	#z_n of size k*1
	return u_d,z_n
	
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

#plt.spy(train_matrix)
#plt.show()

#####################################################
num_given = []
for i in range(train_matrix.shape[1]):
	non_zeros = np.where(train_matrix[:,i] != 0)[0]
	num_given.append(len(non_zeros))

print num_given
avg_num = float(sum(num_given))/float(len(num_given))
print "average number of rated items: " + str(avg_num)


#####################################################

#fill in the missing values by the average of all observed rating for specific item
if(INITILIZATION_ITEM_AVG):
	num_of_items = train_matrix.shape[0]
	avg_item_rating = np.zeros((num_of_items,1))
	for i in range(0,num_of_items):
		avg_item_rating[i]= compute_missing_rating(train_matrix[i,:])

	zero_row_indices = np.where(train_matrix == 0)[0]
	zero_column_indices = np.where(train_matrix == 0)[1]

	zero_item_indices = zip(zero_row_indices,zero_column_indices)

	for zero_item_index in zero_item_indices:
		train_matrix[zero_item_index[0],zero_item_index[1]] = avg_item_rating[zero_item_index[0]]

	print "Missing values filled."

elif(INITILIZATION_USER_AVG):
	num_of_users = train_matrix.shape[1]
	avg_user_rating = np.zeros((num_of_users,1))
	for i in range(0,num_of_users):
		avg_user_rating[i] = compute_missing_rating(train_matrix[:,i])

	zero_row_indices = np.where(train_matrix == 0)[0]
	zero_column_indices = np.where(train_matrix == 0)[1]

	zero_item_indices = zip(zero_row_indices,zero_column_indices)

	for zero_item_index in zero_item_indices:
		train_matrix[zero_item_index[0],zero_item_index[1]] = avg_user_rating[zero_item_index[1]]

	print "Missing values filled."

elif(INITILIZATION_ITEM_USER_MIX):
	num_of_items = train_matrix.shape[0]
	avg_item_rating = np.zeros((num_of_items,1))
	for i in range(0,num_of_items):
		avg_item_rating[i]= compute_missing_rating(train_matrix[i,:])

	zero_row_indices = np.where(train_matrix == 0)[0]
	zero_column_indices = np.where(train_matrix == 0)[1]																																																																																																																																																																																																																																																																		

	zero_item_indices = zip(zero_row_indices,zero_column_indices)

	num_of_users = train_matrix.shape[1]
	avg_user_rating = np.zeros((num_of_users,1))
	for i in range(0,num_of_users):
		avg_user_rating[i] = compute_missing_rating(train_matrix[:,i])

	alphas = np.linspace(0,0.5,endpoint = True,num = 20)

	rmse_validation_alphas = []
	train_matrix_tmp = np.zeros((1000,10000))
	for alpha in alphas:
		for validation_id in validation_ids:
			item = validation_id[0]
			train_matrix_tmp[item[0],item[1]] = alpha*avg_user_rating[item[1]] + (1-alpha)*avg_item_rating[item[0]]
		
		rmse_validation_alphas.append(irmse(train_matrix_tmp,validation_ids))
		print "current alpha check: " + str(alpha) + " with rmse: " + str(irmse(train_matrix_tmp,validation_ids))

	optimal_alpha_index = rmse_validation_alphas.index(min(rmse_validation_alphas))
	optimal_alpha = alphas[optimal_alpha_index]

	print "Optimal alpha is: " + str(optimal_alpha)

	for zero_item_index in zero_item_indices:
		train_matrix[zero_item_index[0],zero_item_index[1]] = optimal_alpha*avg_user_rating[zero_item_index[1]] + (1-optimal_alpha)*avg_item_rating[zero_item_index[0]]

	print "Missing values filled."

elif(INITILIZATION_ITEM_USER_AVG):
	num_of_items = train_matrix.shape[0]
	avg_item_rating = np.zeros((num_of_items,1))
	for i in range(0,num_of_items):
		avg_item_rating[i]= compute_missing_rating(train_matrix[i,:])

	zero_row_indices = np.where(train_matrix == 0)[0]
	zero_column_indices = np.where(train_matrix == 0)[1]																																																																																																																																																																																																																																																																		

	zero_item_indices = zip(zero_row_indices,zero_column_indices)

	num_of_users = train_matrix.shape[1]
	avg_user_rating = np.zeros((num_of_users,1))
	for i in range(0,num_of_users):
		avg_user_rating[i] = compute_missing_rating(train_matrix[:,i])


	rmse_validation= []
	train_matrix_tmp = np.zeros((1000,10000))
	for validation_id in validation_ids:
		item = validation_id[0]
		train_matrix_tmp[item[0],item[1]] = (avg_user_rating[item[1]] + avg_item_rating[item[0]])/2
	
	rmse_validation.append(irmse(train_matrix_tmp,validation_ids))
	print "rmse: " + str(irmse(train_matrix_tmp,validation_ids))

	for zero_item_index in zero_item_indices:
		train_matrix[zero_item_index[0],zero_item_index[1]] = (avg_user_rating[item[1]] + avg_item_rating[item[0]])/2

	print "Missing values filled."

else:	
	print "No imputing needed."




#prediction algorithm
if(SVD_TRUNCATION_SOLUTION):
	U,D,V_transpose = np.linalg.svd(train_matrix,full_matrices = 0,compute_uv = 1)

	D = np.diag(D)

	#plotting the number of eigenvalues selected vs. error rate RMSE
	if(GENERATE_PLOT):
		num_eig_values = matrix_rank(train_matrix)
		rmse_validation = []
		rmse_training = []
		for i in range(1,K+1):
			U_cut = U[:,:i]
			D_cut = D[:i,:i]
			V_cut = V_transpose[:i,:]
			D_sqrt = np.diag(get_square_root(np.diag(D_cut)))
			U_prime = np.dot(U_cut,D_sqrt)
			V_prime = np.dot(D_sqrt,V_cut)
			predicted_matrix = np.dot(U_prime,V_prime)
			rmse_validation.append(irmse(predicted_matrix,validation_ids))
			rmse_training.append(irmse(predicted_matrix,training_ids))																																																																																																																																																																																															
			print i

		plt.plot(rmse_validation,'ro',label = 'validation curve')
		plt.plot(rmse_training,'bo',label = 'training curve')
		plt.xlabel('Number of eigenvalues used')
		plt.ylabel('RMSE')
		plt.title('Error by truncation of matrices')
		plt.legend(bbox_to_anchor=(-1.05, 1), loc=2, borderaxespad=0.)
		plt.show()


		print "Plotting done."																																																																																																																																																																																															

elif(K_MEANS_SOLUTION):
	if(GENERATE_PLOT):
		rmse_validation = []
		for k in range(1,K):
			k_means = KMeans(n_clusters = k, verbose = 0)
			k_means.fit(train_matrix.T)
			labels = k_means.labels_
			print "Struggling with item avg"
			predicted_matrix = get_cluster_item_avg(train_matrix_keep,labels,validation_ids)
			rmse_validation.append(irmse(predicted_matrix,validation_ids))
			print k

		plt.plot(rmse_validation,'ro',label = 'validation curve')
		plt.xlabel('Number of clusters used')
		plt.ylabel('RMSE')
		plt.title('Error by choosing the number of clusters')
		plt.legend(bbox_to_anchor=(-1.05, 1), loc=2, borderaxespad=0.)
		plt.show()

elif(SVD_IMPROVED_SOLUTION):
	#numbers grow to big!!! -> nan in the solution!
	U = np.random.rand(1000,BEST_K)
	Z = np.random.rand(10000,BEST_K)
	Z = Z.T
	j = 0

	for training_id in training_ids:
		nz_item = training_id[0]
		d = nz_item[0]
		n = nz_item[1]
		rating = nz_item[2]
		i = 0
		while(i < EPOCHE):
			U[d,:],Z[:,n] = sgd(rating,U[d,:],Z[:,n],0.001)
			#print U[d,:],Z[:,n]
			i = i + 1

		#print np.dot(U[d,:],Z[:,n]) - rating
		#print rating
		#print np.dot(U[d,:],Z[:,n])
		j = j + 1
		#print j

	print "Optimization done."


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
#use the truncated prediction matrix (obtained by the best k singular values observed from the plot)
if(SVD_TRUNCATION_SOLUTION):
	U_cut = U[:,:BEST_K]
	D_cut = D[:BEST_K,:BEST_K]
	V_cut = V_transpose[:BEST_K,:]
	D_sqrt = np.diag(get_square_root(np.diag(D_cut)))
	U_prime = np.dot(U_cut,D_sqrt)
	V_prime = np.dot(D_sqrt,V_cut)
	prediction_matrix = np.dot(U_prime,V_prime)

	print "The overall RMSE prediction error for selected " + str(BEST_K) + " singular values is: " + str(irmse(prediction_matrix,validation_ids))

elif(BASELINE_SOLUTION):
	print "The overall RMSE prediction error for baseline solution is: " + str(irmse(train_matrix,validation_ids))

elif(K_MEANS_SOLUTION):
	k_means = KMeans(n_clusters = BEST_K, verbose = 0)
	k_means.fit(train_matrix.T)
	labels = k_means.labels_
	print "Struggling with item avg"
	prediction_matrix = get_cluster_item_avg(train_matrix_keep,labels,testing_ids)

elif(SVD_IMPROVED_SOLUTION):
	prediction_matrix = np.dot(U,Z)

	print "The overall RMSE prediction error for selected " + str(BEST_K) + " optimized singular values is: " + str(irmse(prediction_matrix,validation_ids))



#find the predictions
predictions = []
for prediction_index in prediction_indices:
	if(BASELINE_SOLUTION):
		predictions.append(train_matrix[prediction_index[0],prediction_index[1]])
	elif(SVD_TRUNCATION_SOLUTION):
		predictions.append(prediction_matrix[prediction_index[0],prediction_index[1]])
	elif(K_MEANS_SOLUTION):
		predictions.append(prediction_matrix[prediction_index[0],prediction_index[1]])
	elif(SVD_IMPROVED_SOLUTION):
		predictions.append(prediction_matrix[prediction_index[0],prediction_index[1]])

print "Predictions ready."

#generate prediction file
if(GENERATE_SUBMISSION):
	predictions_file = open("submissions/" + SUBMISSION_FILENAME, "wb")
	open_file_object = csv.writer(predictions_file)
	open_file_object.writerow(["Id","Prediction"])
	open_file_object.writerows(zip(test_ids, predictions))
	predictions_file.close()
	print "Prediction file written."