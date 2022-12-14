import numpy as np
from sklearn.base import BaseEstimator
import queue


def entropy(y):  
    """
    Computes entropy of the provided distribution. Use log(value + eps) for numerical stability
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, n_classes)
        One-hot representation of class labels for corresponding subset
    
    Returns
    -------
    float
        Entropy of the provided subset
    """
    EPS = 0.0005

    # YOUR CODE HERE
    prob = np.mean(y, axis=0)
    entropy = -np.sum(prob * np.log(prob + EPS))
    
    return entropy
    
def gini(y):
    """
    Computes the Gini impurity of the provided distribution
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, n_classes)
        One-hot representation of class labels for corresponding subset
    
    Returns
    -------
    float
        Gini impurity of the provided subset
    """

    # YOUR CODE HERE
    prob = np.mean(y, axis=0)
    gini = 1 - np.sum(prob ** 2)
    
    return gini
    
def variance(y):
    """
    Computes the variance the provided target values subset
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, 1)
        Target values vector
    
    Returns
    -------
    float
        Variance of the provided target vector
    """
    
    # YOUR CODE HERE
    mean = np.mean(y)
    size = y.shape[0]
    variance = np.sum((y - mean) ** 2) / size
    
    return variance

def mad_median(y):
    """
    Computes the mean absolute deviation from the median in the
    provided target values subset
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, 1)
        Target values vector
    
    Returns
    -------
    float
        Mean absolute deviation from the median in the provided vector
    """

    # YOUR CODE HERE
    median = np.median(y)
    size = y.shape[0]
    mad_median = np.sum(np.abs(y - median)) / size
    
    
    return mad_median


def one_hot_encode(n_classes, y):
    y_one_hot = np.zeros((len(y), n_classes), dtype=float)
    y_one_hot[np.arange(len(y)), y.astype(int)[:, 0]] = 1.
    return y_one_hot


def one_hot_decode(y_one_hot):
    return y_one_hot.argmax(axis=1)[:, None]

class Vertex:
    """
    Class for searching
    """
    def __init__(self, node, X, index):
        self.node = node
        self.X = X
        self.index = index
    
    def get(self):
        return (self.node, self.X, self.index)


class Node:
    """
    This class is provided "as is" and it is not mandatory to it use in your code.
    """
    def __init__(self, feature_index, threshold, proba=0):
        self.feature_index = feature_index
        self.value = threshold
        self.proba = proba
        self.left_child = None
        self.right_child = None
        
        
class DecisionTree(BaseEstimator):
    all_criterions = {
        'gini': (gini, True), # (criterion, classification flag)
        'entropy': (entropy, True),
        'variance': (variance, False),
        'mad_median': (mad_median, False)
    }

    def __init__(self, n_classes=None, max_depth=np.inf, min_samples_split=2, 
                 criterion_name='gini', debug=False):

        assert criterion_name in self.all_criterions.keys(), 'Criterion name must be on of the following: {}'.format(self.all_criterions.keys())
        
        self.n_classes = n_classes
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion_name = criterion_name

        self.depth = 0
        self.root = None # Use the Node class to initialize it later
        self.debug = debug

        
        
    def make_split(self, feature_index, threshold, X_subset, y_subset):
        """
        Makes split of the provided data subset and target values using provided feature and threshold
        
        Parameters
        ----------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels for corresponding subset
        
        Returns
        -------
        (X_left, y_left) : tuple of np.arrays of same type as input X_subset and y_subset
            Part of the providev subset where selected feature x^j < threshold
        (X_right, y_right) : tuple of np.arrays of same type as input X_subset and y_subset
            Part of the providev subset where selected feature x^j >= threshold
        """

        # YOUR CODE HERE
        
        # Find the index where our data was divided by feature_index at threshold value
        left_index = X_subset[:, feature_index] < threshold
        right_index = X_subset[:, feature_index] >= threshold
        
        # Split the feature X, target y by the index we have already found
        X_left = X_subset[left_index]
        y_left = y_subset[left_index]
        
        X_right = X_subset[right_index]
        y_right = y_subset[right_index]
    
        
        return (X_left, y_left), (X_right, y_right)
    
    def make_split_only_y(self, feature_index, threshold, X_subset, y_subset):
        """
        Split only target values into two subsets with specified feature and threshold
        
        Parameters
        ----------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels for corresponding subset
        
        Returns
        -------
        y_left : np.array of type float with shape (n_objects_left, n_classes) in classification 
                   (n_objects, 1) in regression 
            Part of the provided subset where selected feature x^j < threshold

        y_right : np.array of type float with shape (n_objects_right, n_classes) in classification 
                   (n_objects, 1) in regression 
            Part of the provided subset where selected feature x^j >= threshold
        """

        # YOUR CODE HERE
        
        # Find the index where our data was divided by feature_index at threshold value
        left_index = X_subset[:, feature_index] < threshold
        right_index = X_subset[:, feature_index] >= threshold
        
        # Split the target y by the index we have already found
        y_left = y_subset[left_index]
        y_right = y_subset[right_index]
        
        return y_left, y_right

    def choose_best_split(self, X_subset, y_subset):
        """
        Greedily select the best feature and best threshold w.r.t. selected criterion
        
        Parameters
        ----------
        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels or target values for corresponding subset
        
        Returns
        -------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        """
        # YOUR CODE HERE
        feature_index = 0
        threshold = 0
        min_criterion = np.inf
        
        # We iterate all the thresholds among the feature_index values and choose the best one which gives the minimum criterion value
        # i - feature_index value
        # j - threshold value
        for i in range(X_subset.shape[1]):
            for j in np.unique(X_subset[:, i]):
                y_left, y_right = self.make_split_only_y(i, j, X_subset, y_subset)
                Q = X_subset.shape[0]
                L = y_left.shape[0]
                R = y_right.shape[0]
                if L == 0 or R == 0:
                    continue
                # We use the formula to compute the criterion which was given in the notebook
                current_criterion = L/Q * self.criterion(y_left) + R/Q * self.criterion(y_right)
                
                # Find the smallest criterion value and corresponding threshold as well as feature_index
                if (current_criterion < min_criterion):
                    min_criterion = current_criterion
                    threshold = j
                    feature_index = i
            
            
        return feature_index, threshold
    
    def make_tree(self, X_subset, y_subset, depth):
        """
        Recursively builds the tree
        Parameters
        ----------
        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels or target values for corresponding subset
        
        Returns
        -------
        root_node : Node class instance
            Node of the root of the fitted tree
        """

        # YOUR CODE HERE
        # when you reach the leaves, so you don't need to add left and right part of tree, conditions to end the recursion function
        if depth >= self.max_depth:
            new_node = Node(None, None)   
            if not self.classification:
                new_node.proba = np.mean(y_subset)
            else:
                if y_subset.shape[0] != 0:
                    value = np.sum(y_subset, axis=0) / y_subset.shape[0]
                else:
                    value = 0
                new_node.proba = value  
        else:
            # You are not in the leaves so we need to continue make tree for the left and right subtrees recursively
            feature, threshold = self.choose_best_split(X_subset, y_subset)
            new_node = Node(feature_index=feature, threshold=threshold)
            # Make a tree by threshold value and feature_index
            (X_left, y_left), (X_right, y_right) = self.make_split(new_node.feature_index, new_node.value, X_subset, y_subset)
            # Make tree recursively with left and right subtrees
            new_node.left_child = self.make_tree(X_left, y_left, depth + 1)
            new_node.right_child = self.make_tree(X_right, y_right, depth + 1)
            
        return new_node
        
    def fit(self, X, y):
        """
        Fit the model from scratch using the provided data
        
        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data to train on

        y : np.array of type int with shape (n_objects, 1) in classification 
                   of type float with shape (n_objects, 1) in regression 
            Column vector of class labels in classification or target values in regression
        
        """
        assert len(y.shape) == 2 and len(y) == len(X), 'Wrong y shape'
        self.criterion, self.classification = self.all_criterions[self.criterion_name]
        if self.classification:
            if self.n_classes is None:
                self.n_classes = len(np.unique(y))
            y = one_hot_encode(self.n_classes, y)

        self.root = self.make_tree(X, y, depth=0)
    
    def predict(self, X):
        """
        Predict the target value or class label  the model from scratch using the provided data
        
        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data the predictions should be provided for

        Returns
        -------
        y_predicted : np.array of type int with shape (n_objects, 1) in classification 
                   (n_objects, 1) in regression 
            Column vector of class labels in classification or target values in regression
        
        """

        # YOUR CODE HERE
        if not self.classification:
            # This prediction is for target values in regression
            y_predicted = np.full(X.shape[0], 0)         
            new_q = queue.Queue()
            index = np.arange(X.shape[0])   
            new_q.put(Vertex(self.root, X, index))
            while not new_q.empty():
                vertex = new_q.get()
                current_node, current_X, current_index = vertex.get()
                if current_node.left_child is not None or current_node.right_child is not None:
                    (X_left, y_left), (X_right, y_right) = self.make_split(current_node.feature_index, current_node.value,
                                                                           current_X, current_index)
                    new_q.put(Vertex(current_node.left_child, X_left, y_left))
                    new_q.put(Vertex(current_node.right_child, X_right, y_right))  
                else:
                     y_predicted[current_index] = current_node.proba
                           
        else:
            # This prediction is for class labels in classification
            y_predicted = np.argmax(self.predict_proba(X), axis=1).reshape(X.shape[0])
            
        return y_predicted
        
    def predict_proba(self, X):
        """
        Only for classification
        Predict the class probabilities using the provided data
        
        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data the predictions should be provided for

        Returns
        -------
        y_predicted_probs : np.array of type float with shape (n_objects, n_classes)
            Probabilities of each class for the provided objects
        
        """
        assert self.classification, 'Available only for classification problem'

        # YOUR CODE HERE
        
        # This predict proba is used only for classification problem
        y_predicted_probs = np.full((X.shape[0], self.n_classes), 0, dtype=float)
        new_q = queue.Queue()
        index = np.arange(X.shape[0])
        new_q.put(Vertex(self.root, X, index))
        while not new_q.empty():
            vertex = new_q.get()
            current_node, current_X, current_index = vertex.get()
            if current_node.left_child is not None or current_node.right_child is not None:
                (X_left, y_left), (X_right, y_right) = self.make_split(current_node.feature_index, current_node.value, current_X, current_index)
                new_q.put(Vertex(current_node.left_child, X_left, y_left))
                new_q.put(Vertex(current_node.right_child, X_right, y_right))
            else:
                y_predicted_probs[current_index] = current_node.proba

        return y_predicted_probs
      
