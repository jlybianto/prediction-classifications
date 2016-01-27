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