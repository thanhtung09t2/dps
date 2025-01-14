# DPS: Density Preserving Sampling
This is the implementation of Density Preserving Sampling for a given input dataset

Density Preserving Sampling (DPS) divides the input dataset into a given 
number of folds (2^LEVELS) by maximizing the correntropy between the folds
and can be used as an alternative for cross-validation. The procedure is 
deterministic, so unlike cross-validation it does not need to be repeated.

REFERENCE
Budka, M. and Gabrys, B., 2012. Density Preserving Sampling: Robust and Efficient Alternative to Cross-validation for Error Estimation. IEEE Transactions on Neural Networks and Learning Systems, DOI: 10.1109/TNNLS.2012.2222925. 
