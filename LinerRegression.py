import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import numpy as np
from sklearn import linear_model

# ----------------------------------------------Load dataset
TestScore= np.loadtxt('score.txt')
Passed=np.loadtxt('pass.txt')

# Split the data into training/testing sets
# test set is 16% of data
TestScore_train = TestScore
TestScore_test = TestScore[1:80:10,::]
Passed_train = Passed
Passed_test = Passed[1:80:10]

#---------------------------------------------Liner regression
# Create linear regression object
LR = linear_model.LinearRegression()

# Train the model using the training sets and target
LR.fit(TestScore_train,Passed_train)

# -----------------------------------------------print outputs
# The coefficients
print('Coefficients: \n', LR.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % np.mean((LR.predict(TestScore_test) - Passed_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % LR.score(TestScore_test, Passed_test))

# ------------------------------------------------Plot outputs
# create figure object
fig = plt.figure()
# make figure 3d
ax = fig.gca(projection='3d')
# plot test point
ax.scatter(TestScore_test[::,0],TestScore_test[::,1], Passed_test, color='blue',linewidth=3)
# plot regression line
ax.plot([max(TestScore_test[::,0]),min(TestScore_test[::,0])],[max(TestScore_test[::,1]),min(TestScore_test[::,1])],[0.5,0.5])
ax.legend()
plt.show()