## dataloader.py
Load data from fixed path
## sgd.py
Using GD to predict `U` and `f` simultaneously. This file does not work out well due to the poor approximation `U` right now, so we may need to pend the modification after we get some useful results on using the exact kernel `K` first.
## semi.py
We currently try to get a good result on the semi-supervised setting (to beat `SVM`) first.

This file tries to do semi-supervised learning on target domain test data using target domain [train, test] data only. The solution for prediction `f` is exact. We tune parameters over `gamma` for the kernel, `w_2` for coefficient for regularization term and the number of nearest neighbor to make the kernel sparse.
## full\_data\_semi.py
his file tries to do semi-supervised learning on target domain test data using all data [source\_train, source\_test, source\_para, target\_train, target\_test, target\_para]. We do not complete `K`. The solution for prediction `f` is exact. The tuning parameters `w_2` for coefficient for regularization term. (we use default gamma, which is the sqrt of dimension for each domain)
## Current Problem 
### Kernel
The current RBF Kernel is calculated using Euclidean distance, which might not be a good measurement. (I have gone through some data. Some data points always have large similarity with every point.) This may also cause the ineffectiveness of sparsify the `K` matrix (run `semi.py` for details).
