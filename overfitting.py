# ----------------
# IMPORT PACKAGES
# ----------------

# The pandas package is used to fetch and store data in a DataFrame.
# The matplotlib package is for graphical outputs (eg. box-plot, histogram, QQ-plot).
# The numpy package is for scientific computing and container of generic data (used for generating a continuous distribution)
# The statsmodels is used to find the model coefficients. Formula holds lower case models.
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from matplotlib import colors

# ----------------
# GENERATE DATA
# ----------------

# Set seed for reproducible results
np.random.seed(414)

# Generate data
X = np.linspace(0, 15, 1000)
y = 3 * np.sin(X) + np.random.normal(1 + X, 0.2, 1000)

# Split into training set (70%) and test set (30%)
train_X, train_y = X[:700], y[:700]
test_X, test_y = X[700:], y[700:]

train_df = pd.DataFrame({"X": train_X, "y": train_y})
test_df = pd.DataFrame({"X": test_X, "y": test_y})
df = pd.DataFrame({"X": X, "y": y})

# ----------------
# MODEL TRAINING DATA
# ----------------

# Linear Polynomial Fit of Training Set
poly_line_train = smf.ols(formula="y ~ 1 + X", data=train_df).fit()
print ""
print poly_line_train.summary()
print "Intercept: ", poly_line_train.params[0]
print "Coefficient: ", poly_line_train.params[1]
print "P-Value: ", poly_line_train.pvalues[0]
print "R-Squared: ", poly_line_train.rsquared

# Quadratic Polynomial Fit of Training Set
poly_quad_train = smf.ols(formula="y ~ 1 + X + I(X**2)", data=train_df).fit()
print ""
print poly_quad_train.summary()
print "Intercept: ", poly_quad_train.params[0]
print "Coefficient: ", poly_quad_train.params[1]
print "P-Value: ", poly_quad_train.pvalues[0]
print "R-Squared: ", poly_quad_train.rsquared

# Trigonometric Polynomial Fit of Training Set
poly_trig_train = smf.ols(formula="y ~ X + np.sin(X)", data=train_df).fit()
print ""
print poly_trig_train.summary()
print "Intercept: ", poly_trig_train.params[0]
print "Coefficient: ", poly_trig_train.params[1]
print "P-Value: ", poly_trig_train.pvalues[0]
print "R-Squared: ", poly_trig_train.rsquared

# ----------------
# VISUALIZE TRAINING DATA
# ----------------

# Visualize training set data (approximately 70% of the whole data set) to be used to model.
print ""
plt.figure()
plt.plot(train_df["X"], train_df["y"], "o")
plt.gca().grid(True)
plt.xlabel("X", fontsize=14)
plt.ylabel("Y", fontsize=14)
plt.title("Example of Overfitting - Training Set Data", fontsize=16)
plt.savefig("training_data.png")

# ----------------
# MODEL + VISUALIZE TEST DATA
# ----------------

# Modeling testing set data (30% of the data) with the Linear Polynomial Fit.
predicted_y_line = poly_line_train.predict(test_df["X"])[700:]
resid_line = predicted_y_line - test_df["y"]
mse = sum((predicted_y_line - test_df["y"]) ** 2) / (len(predicted_y_line))
print "Mean Square Error (MSE) of Linear Fit = %s" %mse

plt.figure()
plt.plot(test_df["X"], test_df["y"], "o")
plt.plot(test_df["X"], predicted_y_line, "r")
plt.plot(test_df["X"], resid_line, "g")
plt.gca().grid(True)
plt.xlabel("X", fontsize=14)
plt.ylabel("Y", fontsize=14)
plt.title("Test Set with Linear Polynomial Fit of Training Set", fontsize=16)
plt.savefig("line_fit.png")

# Modeling testing set data (30% of the data) with the Quadratic Polynomial Fit.
predicted_y_quad = poly_quad_train.predict(test_df["X"])[700:]
resid_quad = predicted_y_quad - test_df["y"]
mse = sum((predicted_y_quad - test_df["y"]) ** 2) / (len(predicted_y_quad))
print "Mean Square Error (MSE) of Quadratic Fit = %s" %mse

plt.figure()
plt.plot(test_df["X"], test_df["y"], "o")
plt.plot(test_df["X"], predicted_y_quad, "r")
plt.plot(test_df["X"], resid_quad, "g")
plt.gca().grid(True)
plt.xlabel("X", fontsize=14)
plt.ylabel("Y", fontsize=14)
plt.title("Test Set with Quadratic Polynomial Fit of Training Set", fontsize=16)
plt.savefig("quad_fit.png")

# Modeling testing set data (30% of the data) with the Trigonometric Polynomial Fit.
predicted_y_trig = poly_trig_train.predict(test_df["X"])[700:]
resid_trig = predicted_y_trig - test_df["y"]
mse = sum((predicted_y_trig - test_df["y"]) ** 2) / (len(predicted_y_trig))
print "Mean Square Error (MSE) of Trigonometric Fit = %s" %mse

plt.figure()
plt.plot(test_df["X"], test_df["y"], "o")
plt.plot(test_df["X"], predicted_y_trig, "r")
plt.plot(test_df["X"], resid_trig, "g")
plt.gca().grid(True)
plt.xlabel("X", fontsize=14)
plt.ylabel("Y", fontsize=14)
plt.title("Test Set with Trigonometric Polynomial Fit of Training Set", fontsize=16)
plt.savefig("trig_fit.png")