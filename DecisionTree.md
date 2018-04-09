# Decision Tree

Andrew Ng's Machine Learning Course does not include Decision Tree. I found some of useful materials for these topics from CMU's machine learning course [10-601](http://www.cs.cmu.edu/~ninamf/courses/601sp15/lectures.shtml).  
* Decision Tree

    Example:    

    ![DecisionTree](../images/DecisionTree.jpg)

    * Each internal node: test one discrete-valued attribute x<sub>i</sub>
    * Each branch from a node: select one value for x<sub>i</sub>
    * Each leaf node: predict Y

* Problem setting
    * Set of possible instances X
        * Each instance x in X is a feature vector
    * Unknown target function f: X &rarr; Y
    * Set of function hypothesis H = {h|h: X &rarr; Y}
        * Each h is a decision tree
        * Tree sorts x to leaf, which assigns y
    * Input: {x<sup>(i)</sup>, y<sup>(i)</sup>) to train unknown target function f
    * Output: h &isin; H that best approximates target function f

* Logic operations:
    * y = x1 &times; x2

    ![AndOper](../images/AndOper.jpg)

    * y = x1 U x2

    ![OrOper](../images/OrOper.jpg)

    * We can actually represent all discrete-wise functions with decision trees

* Instruction
    
    Node = root
    
    Main loop:
    
    1. A &larr; the "best" decision attribute for next node
    1. Assign A as decision attribute for node
    1. For each value of A, create new descendant of node
    1. Sort training examples to leaf nodes
    1. If the training perfectly classified, stop. Else iterate over new leaf nodes

* How to choose attributes: sample entropy

    The entropy of a sample is defined as:

    <img src="http://latex.codecogs.com/svg.latex?H(x)=-\sum_{i=1}^{n}P(x=i)\textrm{log}P(x=i)"/>

    From information theory, H(x) is the expected number of bits needed to encode a random drawn value of x (under most efficient code)

    * Specific conditional entropy:
    
    <img src="http://latex.codecogs.com/svg.latex?H(x|y=v)=-\sum_{i=1}^{n}P(x=i|y=v)\textrm{log}P(x=i|y=v)"/>

    * Conditional entropy:

    <img src="http://latex.codecogs.com/svg.latex?H(x|y)=-\sum_{v{\in}values(y)}P(y=v)H(x|y=v)

    * Mutual information (Information Gain) of x and y

    <img src="http://latex.codecogs.com/svg.latex?l(x, y)=H(x)-H(x|y)=H(y)-H(y|x)"/>

* Function approximation: the big picture

    ![FunctionApproximation](../images/FunctionApproximation.jpg)

    f: X &rarr; Y, suppose we have 20 training sets

    # of trees: 2<sup>20</sup>

    # of teachable functions decision trees can represent = 2 <sup>2<sup>20</sup></sup>

    * There are many decision trees that can match well with training examples, but don't agree on unseen examples

    * Usually assumptions: grow the tree just big enough to fit the training data

    * "Occam's razor": prefer the simplest hypothesis that fits the data

* Overfitting in Decision Tree

    ![DT_overfitting](../images/DT_overfitting.jpg)

* Reduce-Error Pruning
    
    Split data into training and cross-validation set

    Create tree that classifies training set correctly

    Do until futher pruning is harmful

        1. Evaluate impact on cross-validation set of pruning each possible node
        2. Greedily remove the one that mostly improves validation set accuracy

* 2 core aspects of machine learning

    * Algorithm desing
    * Confidence bounds, Generalization

* Generalization of decision trees:

    If we were able to find a small decision tree that explains data well, then good generalization garuantees

    * NP hard
    * Very nice practical heuristics: top down algorithms, e.g. ID3
    * Why not use split according to error rate
        * May get stuck in local minimal
    * If measure of progress is entropy, we can always garauntee success under some formal relationship between the class of splits and the targets
