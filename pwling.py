"""
    PWLING: pairwise causality measures in linear non-Gaussian model
    Version 1.2, Aapo Hyvarinen, Feb 2013
    Input: Data matrix with variables as rows, and index of method [1...5]
    Output: Matrix LR with likelihood ratios
        If entry (i,j) in that matrix is positive, 
        estimate of causal direction is i -> j
    Methods 1...5 are as follows:
      1: General entropy-based method, for variables of any distribution
      2: First-order approximation of LR by tanh, for sparse variables
      3: Basic skewness measure, for skewed variables
      4: New skewness-based measure, robust to outliers
      5: Dodge-Rousson measure, for skewed variables
      If you want to use method 3 or 4 without skewness correction, 
         input -3 or -4 as the method.
    See http://www.cs.helsinki.fi/u/ahyvarin/code/pwcausal/ for more information
"""

import math
import numpy as np
from scipy.stats import skew

def mentappr(x):
    """
        MENTAPPR: Compute MaxEnt approximation of negentropy and differential entropy
        Based on NIPS*97 paper, www.cs.helsinki.fi/u/ahyvarin/papers/NIPS97.pdf
        Input: sample of continous-valued random variable as a vector. 
            For matrices, entropy computed for each column
        Output: (differential) entropy and, optionally, negentropy
    """
    # Standardize
    x=x-np.mean(x)
    xstd=np.std(x, ddof=1)
    x=x/xstd
    #Constants we need
    k1=36/(8*math.sqrt(3)-9)
    gamma=0.37457
    k2=79.047
    gaussianEntropy=math.log(2*math.pi)/2+1/2
    #This is negentropy
    negentropy = k2*((np.mean(np.log(np.cosh(x)))-gamma)**2)+k1*(np.mean(x*np.exp(-(x**2)/2))**2)
    #This is entropy
    entropy = gaussianEntropy - negentropy + math.log(xstd)
    return entropy

def case1(n, X, C, m):
    #General entropy-based method, for variables of any distribution
    #Initialize output matrix
    LR=np.zeros((n,n))
    #Loop through pairs
    for i in range(n):
        for j in range(n):
            if i != j:
                res1= X[j]-C[j][i]*X[i]
                res2=X[i]-C[i][j]*X[j]
                LR[i][j]=mentappr(X[j])-mentappr(X[i])-mentappr(res1)+mentappr(res2)
    return LR

def case2(n, X, C, m):
    #first-order approximation of LR by tanh, for sparse variables
    LR=C *(np.dot(X,np.tanh(X.T))-np.dot(np.tanh(X),X.T))/m
    return LR
def case3(n, X, C, m):
    #basic skewness measure, for skewed variables
    LR=C*(-np.dot(X,(X.T**2))+np.dot(X**2,X.T))/m
    return LR
def case4(n, X, C, m):
    #New skewed measure, robust to outliers
    gX=np.log(np.cosh(np.maximum(X,0)))
    LR= C*(-(np.dot(X,gX.T)/m)+(np.dot(gX,X.T)/m))
    return LR
def case5(n, X, C, m):
    #Dodge-Rousson measure, for skewed variables
    LR=(-((np.dot(X,X.T**2))**2)+(np.dot(X**2,X.T))**2)/m
    return LR

def pwling(X, method):
    #Get size parameters
    [n,m]=X.shape
    #Standardize each variable
    X=(X.T-X.T.mean(axis=0)).T
    dividingValue = X.T.std(axis=0, ddof=1)+np.spacing(1)
    X=(X.T/dividingValue).T
    #If using skewness measures with skewness correction, make skewnesses positive
    if method==3 or method==4:
        for i in range(n): 
            skewness = skew(X[i], axis=0)
            if skewness != 0:
                signOfSkewness = skewness/math.fabs(skewness)
                X[i]=X[i]*signOfSkewness        
    #Compute covariance matrix
    C = np.cov(X)
    #Compute causality measures
    cases = [case1, case2, case3, case4, case5]
    return cases[abs(method)-1](n, X, C, m)
        
def doAllPwling(X):
    finalList = []
    for i in range(1,6):
        result = pwling(X, i)
        finalList.append(result[0][1])
    return np.array(finalList)

def doPwling(X, method):
    result = pwling(X, method)
    return result[0][1]
