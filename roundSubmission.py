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

INPUT = "bestSubmission"

test_ids = []
predictions = []
with open("submissions/"+INPUT+".csv","rb") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        test_ids.append(row['Id'])
        predictions.append(round(float(row['Prediction'])))

predictions_file = open("submissions/" + INPUT +"_rounded.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["Id","Prediction"])
open_file_object.writerows(zip(test_ids, predictions))
predictions_file.close()
print "Prediction file written."