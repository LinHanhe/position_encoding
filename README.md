# Position encoding

This code implements a deep learning for motorcycle helmet use classification.


There are two files with different training schemes. 
The first one (Helmet_use_classification.ipynb) train a multi-class deep model with softmax cross-entropy loss, which suffers from class imbalance issue. 
The second (Helmet_use_classification_encode.ipynb) uses a new scheme and transform the class imbalance multi-class classification task into 10 binary classification problems. 

To run both files, you need to download the HELMET dataset from https://osf.io/4pwj8/.

