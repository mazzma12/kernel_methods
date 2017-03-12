# kernel_methods
This repository was created jointly with Konstantin Mischenko for the project of the course Machine Learning with Kernel Methods taught by J-P Vert and J. Mairal at ENS Cachan. The goal was to classify images, using kernel methods, but no machine learning, nor computer vision libraries. 
The src file contains 2 scripts : 
- SVM.py which contains an implementation of a Multiclass SVM, using a Quadratic Program solver to solve the dual problem for SVM.
And some basic Machine learning algorithms equivalent to CVGridSearch in scikit-learn
- data_augmentation.py which mainly contains an implementation of Scharr Gradient and Histogram of Gradients, used for preprocessing the data. Other functions are also useful to visualize the images and the perform data augmentation (flipping, shifting, ...)
