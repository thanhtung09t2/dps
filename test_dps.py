import numpy as np
import os
from dps_func import dps

# File and data handling
data_file = "yeast.csv"
M = np.loadtxt(data_file, delimiter=",")  # Replace with appropriate delimiter if not ','

r, c = M.shape
A = M[:, :c - 1]
LABS = M[:, c - 1]

LEVELS = 3

M = np.column_stack((A, LABS))
R, H = dps(A, LEVELS, LABS)

max_fold = 2 ** LEVELS
Fold0 = M[R == 0, :]
Fold1 = M[R == 1, :]
Fold2 = M[R == 2, :]
Fold3 = M[R == 3, :]
Fold4 = M[R == 4, :]
Fold5 = M[R == 5, :]
Fold6 = M[R == 6, :]
Fold7 = M[R == 7, :]
Fold8 = M[R == 8, :]


# File writing
filepath, name = os.path.split(data_file)
name, ext = os.path.splitext(name)

file_fold0 = f"{name}_dps_remaining_samples.csv"
file_fold1 = f"{name}_dps_1.csv"
file_fold2 = f"{name}_dps_2.csv"
file_fold3 = f"{name}_dps_3.csv"
file_fold4 = f"{name}_dps_4.csv"
file_fold5 = f"{name}_dps_5.csv"
file_fold6 = f"{name}_dps_6.csv"
file_fold7 = f"{name}_dps_7.csv"
file_fold8 = f"{name}_dps_8.csv"

np.savetxt(file_fold0, Fold0, delimiter=",", fmt="%.6f")
np.savetxt(file_fold1, Fold1, delimiter=",", fmt="%.6f")
np.savetxt(file_fold2, Fold2, delimiter=",", fmt="%.6f")
np.savetxt(file_fold3, Fold3, delimiter=",", fmt="%.6f")
np.savetxt(file_fold4, Fold4, delimiter=",", fmt="%.6f")
np.savetxt(file_fold5, Fold5, delimiter=",", fmt="%.6f")
np.savetxt(file_fold6, Fold6, delimiter=",", fmt="%.6f")
np.savetxt(file_fold7, Fold7, delimiter=",", fmt="%.6f")
np.savetxt(file_fold8, Fold8, delimiter=",", fmt="%.6f")