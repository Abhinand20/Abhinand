#Decision Tree classifier from scratch

class Node():
    def __init__(self,feature_index=None,threshold=None,left=None,right=None,info_gain=None,value=None):
        '''Class used to represent a Node - left and right are also Nodes (Creates a tree)'''
        
        #For decision node
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain
        
        #For leaf node
        self.value = value
  
  class CustomDecisionTreeClassifier():
    def __init__(self,min_samples_split=2,max_depth = 2):
        #Initialize root node of the tree
        self.root = None
        
        #Initialize stopping conditions
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
    
    def build_tree(self,dataset,curr_depth=0):
        '''Build tree recursively'''
        
        X , Y = dataset[:,:-1],dataset[:,-1]
        num_samples , num_features = np.shape(X)
        
        #If end condition is not met - Build and return the Decision Nodes of the tree
        if num_samples >= self.min_samples_split and curr_depth <= self.max_depth:
            
            best_split = self.get_best_split(dataset,num_samples,num_features)
            #If info_gain = 0 ; Perfect split - return Leaf Node otherwise keep building decision nodes
            if best_split['info_gain'] > 0:
                #Build left subtree recursively
                left_subtree = self.build_tree(best_split['dataset_left'],curr_depth + 1)
                #Build right subtree recursively
                right_subtree = self.build_tree(best_split['dataset_right'],curr_depth + 1)
                #Return Decison node for right and left subtrees
                return Node(best_split['feature_index'],best_split['threshold'],left_subtree,right_subtree,best_split['info_gain'])
            
        #If end condition is met - We have reached leaf node so return the mode of predicted values
        return Node(value=self.calculate_leaf_value(Y))
    
    def get_best_split(self,dataset,num_samples,num_features):
        best_split = {}
        max_info_gain = -float('inf') #To find the max info gain
        
        for feature_idx in range(num_features):
            feature_values = dataset[:,feature_idx] 
            possible_thresholds = np.unique(feature_values) #Considering all unique feature values as threshold
            
            for threshold in possible_thresholds:
                dataset_left,dataset_right = self.split(dataset,feature_idx,threshold)
                #Extra check to avoid division by 0
                #Check if childs are not null
                if len(dataset_left) > 0 and len(dataset_right) > 0:
                    #Get the values used to calc IG
                    y,left_y,right_y = dataset[:,-1],dataset_left[:,-1],dataset_right[:,-1]
                    curr_info_gain = self.calc_info_gain(y,left_y,right_y)
                    
                    if curr_info_gain > max_info_gain:
                        
                        best_split['feature_index'] = feature_idx
                        best_split['threshold'] = threshold
                        best_split['dataset_left'] = dataset_left
                        best_split['dataset_right'] = dataset_right
                        best_split['info_gain'] = curr_info_gain
                        
                        max_info_gain = curr_info_gain
        
    
        return best_split
    
    
    def split(self,dataset,feature_idx,threshold):
        
        dataset_left = np.array([row for row in dataset if row[feature_idx] < threshold])
        dataset_right = np.array([row for row in dataset if row[feature_idx] >= threshold])
        return dataset_left, dataset_right
    
    def calc_info_gain(self,parent,left_child,right_child):
        
        weight_l = len(left_child)/len(parent)
        weight_r = len(right_child)/len(parent)
        
        IG = self.gini_index(parent) - ( weight_l*self.gini_index(left_child) + weight_r*self.gini_index(right_child))
        
        return IG
                                        
    def gini_index(self,y):
        labels = np.unique(y)
        gini = 0
        for label in labels:
            p_label = len(y[y==label]) / len(y)
            gini += p_label**2
        
        return 1 - gini
        
    def calculate_leaf_value(self,Y):
        return max(list(Y),key=list(Y).count)
    
    
    def print_tree(self, tree=None, indent=" "):
        ''' function to print the tree '''
        
        if not tree:
            tree = self.root

        if tree.value is not None:
            print(tree.value)

        else:
            print("X_"+str(tree.feature_index), "<=", tree.threshold, "?", tree.info_gain)
            print("%sleft:" % (indent), end="")
            self.print_tree(tree.left, indent + indent)
            print("%sright:" % (indent), end="")
            self.print_tree(tree.right, indent + indent)
    
    def fit(self, X, Y):
        ''' function to train the tree '''
        
        dataset = np.concatenate((X, Y), axis=1)
        self.root = self.build_tree(dataset)
    
    def predict(self, X):
        ''' function to predict new dataset '''
        
        preditions = [self.make_prediction(x, self.root) for x in X]
        return preditions
    
    def make_prediction(self, x, tree):
        ''' function to predict a single data point '''
        
        if tree.value!=None: return tree.value
        feature_val = x[tree.feature_index]
        if feature_val < tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)
