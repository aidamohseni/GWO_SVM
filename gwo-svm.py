import random
import time
import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import benchmarks

data = pd.read_csv("cleavland.csv")
data.head()

X = data.iloc[:, :13].values
Y = data['RESULT'].values
print(X.shape)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.40, random_state=14)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Applying SVM
svcclassifier = SVC(kernel='poly', random_state=50)
svcclassifier.fit(X_train, y_train)
y_pred = svcclassifier.predict(X_test)
P = accuracy_score(y_pred, y_test)
print("Accuracy score for SVM:", P)


# GWO Optimization algo

def GWO(object, lb, ub, dimension, SearchAgentsNumber, Max_iter):

    Alpha_pos = np.zeros(dimension)
    Alpha_score = float("inf")

    Beta_pos = np.zeros(dimension)
    Beta_score = float("inf")

    Delta_pos = np.zeros(dimension)
    Delta_score = float("inf")

    if not isinstance(lb, list):
        lb = [lb] * dimension
    if not isinstance(ub, list):
        ub = [ub] * dimension

    # Initialize the positions of search agents
    Positions = np.zeros((SearchAgentsNumber, dimension))
    for i in range(dimension):
        Positions[:, i] = np.random.uniform(0, 1, SearchAgentsNumber) * (ub[i] - lb[i]) + lb[i]

    Convergence_curve = np.zeros(Max_iter)

    # Main loop
    for l in range(0, Max_iter):
        for i in range(0, SearchAgentsNumber):


            for j in range(dimension):
                Positions[i, j] = np.clip(Positions[i, j], lb[j], ub[j])


            fitness = object(Positions[i, :])


            if fitness < Alpha_score:
                Alpha_score = fitness;  # Update alpha
                Alpha_pos = Positions[i, :].copy()

            if (fitness > Alpha_score and fitness < Beta_score):
                Beta_score = fitness  # Update beta
                Beta_pos = Positions[i, :].copy()

            if (fitness > Alpha_score and fitness > Beta_score and fitness < Delta_score):
                Delta_score = fitness  # Update delta
                Delta_pos = Positions[i, :].copy()

        a = 2 - l * ((2) / Max_iter);  # a decreases linearly fron 2 to 0


        for i in range(0, SearchAgentsNumber):
            for j in range(0, dimension):
                r1 = random.random()
                r2 = random.random()

                A1 = 2 * a * r1 - a;  # Equation (3.3)
                C1 = 2 * r2;  # Equation (3.4)

                D_alpha = abs(C1 * Alpha_pos[j] - Positions[i, j]);  # Equation (3.5)-part 1
                X1 = Alpha_pos[j] - A1 * D_alpha;  # Equation (3.6)-part 1

                r1 = random.random()
                r2 = random.random()

                A2 = 2 * a * r1 - a;
                C2 = 2 * r2;

                D_beta = abs(C2 * Beta_pos[j] - Positions[i, j]);
                X2 = Beta_pos[j] - A2 * D_beta;

                r1 = random.random()
                r2 = random.random()

                A3 = 2 * a * r1 - a;
                C3 = 2 * r2;

                D_delta = abs(C3 * Delta_pos[j] - Positions[i, j]);
                X3 = Delta_pos[j] - A3 * D_delta;

                Positions[i, j] = (X1 + X2 + X3) / 3
        Convergence_curve[l] = Alpha_score;


    print(Positions.shape)
    print("Alpha position=", Alpha_pos);
    print("Beta position=", Beta_pos);
    print("Delta position=", Delta_pos);
    return Alpha_pos, Beta_pos;


# Setting GWO parameters#######################
iters = 100
wolves = 5
dimension = 13
search_domain = [0, 1]
lb = -1.28
ub = 1.28
colneeded = [0, 1, 2, 4, 5, 7, 8, 10, 11]
modified_data = pd.DataFrame()
for i in colneeded:
    modified_data[data.columns[i]] = data[data.columns[i]].astype(float)
func_details = benchmarks.getFunctionDetails(6)

for i in range(0, 10):
    alpha, beta = GWO(getattr(benchmarks, 'F7'), lb, ub, dimension, wolves, iters)
