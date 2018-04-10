class Node(object):
    def __init__(self, groups):
        '''Initialized a node, it has left child and right child, cutoff value,
        index for feature column, and terminal value if it's a leaf node.
        If a node only has one label, make it a leaf node.
        
        Input:
            groups: a list of examples
        '''
        self.groups = groups
        self.left = None
        self.right = None
        self.value = None
        self.index = None
        self.terminal = None
        if len(Node.get_labels(groups)) == 1:
            self.to_terminal()
    def get_size(self):
        '''Return the size of examples in current node'''
        return len(self.groups)
    @staticmethod
    def get_labels(groups):
        '''Return a list of unique labels'''
        return list(set([item[-1] for item in groups]))
    @staticmethod
    def gini(groups):
        '''Calculate the gini index of current node
        gini = 1-sum(p(i)**2), i in all labels
        '''
        if not groups:
            return 0
        gini = 1
        for label in Node.get_labels(groups):
            gini -= ([item[-1] for item in groups].count(label)/len(groups))**2
        return gini
    def split(self):
        '''Try different splits and pick the split with the smallest gini index'''
        gini = 1
        # O(n^2), can we do better?
        for index in range(len(self.groups[0])-1):
            for value in [item[index] for item in self.groups]:
                leftlist, rightlist = self.split_groups(index, value)
                gini_c = (len(leftlist)*Node.gini(leftlist)+len(rightlist)*Node.gini(rightlist))/len(self.groups)
                if gini_c < gini:
                    gini = gini_c
                    self.value = value
                    self.index = index
                    # this may creates a lot of unused Node objects
                    self.left = Node(leftlist)
                    self.right = Node(rightlist)
    def split_groups(self, index, value):
        '''Split the group by the index and value
        
        Input:
            index: feature column
            value: cutoff value; example with feature column smaller than this will
        assigned to left child, otherwise right child.
        
        Return:
            list of examples for left child and right child
        '''
        leftlist, rightlist = [], []
        for example in self.groups:
            if example[index] < value:
                leftlist.append(example)
            else:
                rightlist.append(example)
        return leftlist, rightlist
    def to_terminal(self):
        '''Save this node as terminal leaf
        '''
        labels = Node.get_labels(self.groups)
        # might have a problem when the counts are the same
        self.terminal = max(labels, key = labels.count)
    def get_ind_val(self):
        '''Return the index and value'''
        return self.index, self.value
    def get_terminal(self):
        '''Return the classification result'''
        return self.terminal

class CART_tree(object):
    def __init__(self):
        '''Initialize a tree with root node, max_depth and min_size'''
        self.root = None
        self.max_depth = None
        self.min_size = None
    def train(self, training, max_depth = 1, min_size = 1):
        '''Train the tree with training examples
        
        Input:
            max_depth, int
            min_size, int
        '''
        self.root = Node(training)
        self.max_depth = max_depth
        self.min_size = min_size
        self.rec_split(self.root, 0)
    def rec_split(self, node, depth):
        '''Recursively split the node, until meet either of the following conditions:
        1. Max tree depth or minimum node size reached.
        2. The node is leaf node that only has one label.
        
        Input:
            node: Node object
            depth: int
        '''
        if depth == self.max_depth or node.get_size() <= self.min_size:
            # reach the max_depth or min_size, return
            node.to_terminal()
            return
        if node.get_terminal() is not None:
            # leaf node that only contains one label, return
            return
        # after previous 2 cases, the remaining nodes have group size larger than 2
        # and have mixed labels
        node.split()
        self.rec_split(node.left, depth+1)
        self.rec_split(node.right, depth+1)
        return
    def predict(self, groups):
        '''Given a list of groups, predict their labels
        
        Input:
            groups: list, input features
        Return:
            preds: list, predicted labels
        '''
        preds = []
        for group in groups:
            preds.append(CART_tree._predict(group, self.root))
        return preds
    @staticmethod
    def _predict(group, node):
        '''Recursively assign the input feature to one of the childs of 
        current node, until it finds a leaf node.
        
        Input:
            group: list, input feature
            node: Node object
        Return:
            terminal value of a leaf node
        '''
        if node.get_terminal() is not None:
            return node.get_terminal()
        index, value = node.get_ind_val()
        if group[index] < value:
            return CART_tree._predict(group, node.left)
        else:
            return CART_tree._predict(group, node.right)
