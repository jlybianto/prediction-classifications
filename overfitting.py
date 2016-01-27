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

# Linear Polynomial Fit
poly_line = smf.ols(formula="y ~ 1 + X", data=train_df).fit()
print ""
print poly_line.summary()
print "Intercept: ", poly_line.params[0]
print "Coefficient: ", poly_line.params[1]
print "P-Value: ", poly_line.pvalues[0]
print "R-Squared: ", poly_line.rsquared

# Quadratic Polynomial Fit
poly_quad = smf.ols(formula="y ~ 1 + X + I(X**2)", data=train_df).fit()
print ""
print poly_quad.summary()
print "Intercept: ", poly_quad.params[0]
print "Coefficient: ", poly_quad.params[1]
print "P-Value: ", poly_quad.pvalues[0]
print "R-Squared: ", poly_quad.rsquared