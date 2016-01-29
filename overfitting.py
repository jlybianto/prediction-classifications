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

# ----------------
# MODEL DATA
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

# Linear Polynomial Fit of Testing Set
poly_line_test = smf.ols(formula="y ~ 1 + X", data=test_df).fit()
print ""
print poly_line_test.summary()
print "Intercept: ", poly_line_test.params[0]
print "Coefficient: ", poly_line_test.params[1]
print "P-Value: ", poly_line_test.pvalues[0]
print "R-Squared: ", poly_line_test.rsquared

# Quadratic Polynomial Fit of Testing Set
poly_quad_test = smf.ols(formula="y ~ 1 + X + I(X**2)", data=test_df).fit()
print ""
print poly_quad_test.summary()
print "Intercept: ", poly_quad_test.params[0]
print "Coefficient: ", poly_quad_test.params[1]
print "P-Value: ", poly_quad_test.pvalues[0]
print "R-Squared: ", poly_quad_test.rsquared



# ----------------
# VISUALIZE DATA
# ----------------

space = np.arange(0, 15, 0.5)
plt.figure(figsize=(10, 10))
plt.scatter(train_df["X"], train_df["y"], alpha=0.5, color="blue")
plt.scatter(test_df["X"], test_df["y"], alpha=0.5, color="red")
plot_line_train, = plt.plot(poly_line_train.params[0] + poly_line_train.params[1] * space, color="darkblue", label="Linear Fit of Train Set")
plt.legend(handles=[plot_line_train], loc=2, fontsize=14)
plt.gca().grid(True)
plt.xlabel("X", fontsize=14)
plt.ylabel("Y", fontsize=14)
plt.xlim(0, 15)
plt.title("Example of Overfitting", fontsize=16)
plt.savefig("overfitting.png")