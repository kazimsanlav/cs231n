[TOC]

# Cs231n Lecture Notes

## Image Classification

### Challenges

- View point variation
- Scale variation
- Deformation
- Occlusion (reduced visibility due to other objects)
- Illumination
- Background clutter (objects may blend into background)
- Intra-class variation

### Classification Pipeline

1. Input
2. Learning
3. Evaluation

### Nearest Neighbor Classifier

We can represent images as I~1~ and I~2~ , then we can use **L1 distance** to compare them:

<a href="https://www.codecogs.com/eqnedit.php?latex=$$&space;d_1&space;(I_1,&space;I_2)&space;=&space;\sum_{p}&space;\left|&space;I^p_1&space;-&space;I^p_2&space;\right|&space;$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$&space;d_1&space;(I_1,&space;I_2)&space;=&space;\sum_{p}&space;\left|&space;I^p_1&space;-&space;I^p_2&space;\right|&space;$$" title="$$ d_1 (I_1, I_2) = \sum_{p} \left| I^p_1 - I^p_2 \right| $$" /></a>

Sum is taken over all pixels. 

We should flatten all images before using them.

```python
# flatten out all images to be one-dimensional
Xtr_rows = Xtr.reshape(Xtr.shape[0], -1) # Xtr_rows becomes 50000 x 3072
Xte_rows = Xte.reshape(Xte.shape[0], -1) # Xte_rows becomes 10000 x 3072
```

Then we can train and evaluate our classifier:

```python
nn = NN() # create a Nearest Neighbor classifier
nn.train(Xtr_rows , Ytr) # training classifier
Yte_predict = nn.predict(Xte_rows) # make prediction on test data
print('accuracy: %f' % (np.mean(Yte_predict == Yte))) # evaluate its performance
```

Following is the NN class implemented in python:

```python

import numpy as np

class NN(object):
    def __init__(self):
        pass
    
def train(self, X , y):
    '''
    X in N*D where each row is an example. Y is 1-dimension of size N
    '''
    # NN will simply remember all the training data
    self.Xtr = X
    self.ytr = y
    
def predict(self, X):
    '''
    X in N*D where each row is an example we want to predict label for
    '''
    num_test = X.shape[0]
    # output type match the input type
    Ypred = np.zeros(num_test, dtype = self.ytr.dtype) 
    
    # loop over all test rows
    for i in range(num_test):
        distances = np.sum(np.abs(self.Xtr - X[i,:]), axis = 1)
        max_index = np.argmin(distances)
        Ypred[i] = sel.ytr[min_index]
    
    return(Ypred)
    
```

We can also use L~2~ distance.

<a href="https://www.codecogs.com/eqnedit.php?latex=$$&space;d_2(I_1,&space;I_2)&space;=&space;\sqrt{\sum_{p}(I_1^2&space;-&space;I_2^2)^2}&space;$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$&space;d_2(I_1,&space;I_2)&space;=&space;\sqrt{\sum_{p}(I_1^2&space;-&space;I_2^2)^2}&space;$$" title="$$ d_2(I_1, I_2) = \sqrt{\sum_{p}(I_1^2 - I_2^2)^2} $$" /></a>

We just need to change a line in our code:

```python
distances = np.sqrt(np.sum(np.square(self.Xtr - X[i,:]), axis = 1))
```

In practice we don't need to include $sqrt()$ as it is a monotonic function and it won't affect the result.

### K-Nearest Neighbor 

If use higher k value, it will help classifier to get generalization by smoothing the boundary.

### Validation sets for Hyperparameter tuning

Implementation on code:

```python
# assume we have Xtr_rows, Ytr, Xte_rows, Yte as before
# recall Xtr_rows is 50,000 x 3072 matrix
Xval_rows = Xtr_rows[:1000, :] # take first 1000 for validation
Yval = Ytr[:1000]
Xtr_rows = Xtr_rows[1000:, :] # keep last 49,000 for train
Ytr = Ytr[1000:]

# find hyperparameters that work best on the validation set
validation_accuracies = []
for k in [1, 3, 5, 10, 20, 50, 100]:
    # use a particular value of k and evaluation on validation data
    nn = NearestNeighbor()
    nn.train(Xtr_rows, Ytr)
    # here we assume a modified NearestNeighbor class that can take a k as input
    Yval_predict = nn.predict(Xval_rows, k = k)
    acc = np.mean(Yval_predict == Yval)
    print 'accuracy: %f' % (acc,)
    
    # keep track of what works on the validation set
    validation_accuracies.append((k, acc))
```



### Summary: Applying kNN in practice

If you wish to apply kNN in practice (hopefully not on images, or perhaps as only a baseline) proceed as follows:

1. Preprocess your data: *Normalize the features* in your data (e.g. one pixel in images) to have <u>zero mean and unit variance</u>. We will cover this in more detail in later sections, and chose not to cover data normalization in this section because pixels in images are usually homogeneous and do not exhibit widely different distributions, alleviating the need for data normalization.
2. If your data is very high-dimensional, consider using a dimensionality reduction technique such as PCA ([wiki ref](http://en.wikipedia.org/wiki/Principal_component_analysis), [CS229ref](http://cs229.stanford.edu/notes/cs229-notes10.pdf), [blog ref](http://www.bigdataexaminer.com/understanding-dimensionality-reduction-principal-component-analysis-and-singular-value-decomposition/)) or even [Random Projections](http://scikit-learn.org/stable/modules/random_projection.html).
3. Split your training data randomly into train/val splits. As a rule of thumb, between 70-90% of your data usually goes to the train split. This setting depends on how many hyperparameters you have and how much of an influence you expect them to have. If there are many hyperparameters to estimate, you should err on the side of having larger validation set to estimate them effectively. If you are concerned about the size of your validation data, it is best to split the training data into folds and perform cross-validation. If you can afford the computational budget it is always safer to go with cross-validation (the more folds the better, but more expensive).
4. Train and evaluate the kNN classifier on the validation data (for all folds, if doing cross-validation) for many choices of **k** (e.g. the more the better) and across different distance types (L1 and L2 are good candidates)
5. If your kNN classifier is running too long, consider using an Approximate Nearest Neighbor library (e.g. [FLANN](http://www.cs.ubc.ca/research/flann/)) to accelerate the retrieval (at cost of some accuracy).
6. Take note of the hyperparameters that gave the best results. There is a question of whether you should use the full training set with the best hyperparameters, since the optimal hyperparameters might change if you were to fold the validation data into your training set (since the size of the data would be larger). In practice it is cleaner to not use the validation data in the final classifier and consider it to be *burned* on estimating the hyperparameters. Evaluate the best model on the test set. Report the test set accuracy and declare the result to be the performance of the kNN classifier on your data.





