"""
    Jonas Peters, Dominik Janzing, Bernhard Schoelkopf (2010): Identifying Cause and Effect on Discrete Data using Additive Noise Models, 
    in Y.W. Teh and M. Titterington (Eds.), Proceedings of The Thirteenth International Conference on Artificial Intelligence and Statistics (AISTATS) 2010, 
    JMLR: W&CP 9, pp 597-604, Chia Laguna, Sardinia, Italy, May 13-15, 2010,
    
        This file is part of discrete_anm.
    
        discrete_anm is free software: you can redistribute it and/or modify
        it under the terms of the GNU General Public License as published by
        the Free Software Foundation, either version 3 of the License, or
        (at your option) any later version.
    
        discrete_anm is distributed in the hope that it will be useful,
        but WITHOUT ANY WARRANTY; without even the implied warranty of
        MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
        GNU General Public License for more details.
    
        You should have received a copy of the GNU General Public License
        along with discrete_anm.  If not, see <http://www.gnu.org/licenses/>. 
"""
from scipy.stats import chi2_contingency
import numpy as np
import math
from scipy.stats.distributions import chi2
def hist3(para, perp, rpbins, pibins):
    """
        This is not exactly the hist3 function of matlab. However, 
        it will produce the same result for our use.
    """
    rbins = np.empty(len(rpbins)+1)
    rbins[:len(rpbins)] = rpbins
    rbins[len(rpbins)] = rpbins[-1]+1
    pbins = np.empty(len(pibins)+1)
    pbins[:len(pibins)] = pibins
    pbins[len(pibins)] = pibins[-1]+1
    finalbins = [rbins, pbins]
    hist, _ = np.histogramdd([para, perp], bins=finalbins)
    return hist

def chiSqQuant1(x, y, num_states_x, num_states_y):
    if num_states_x == 1 or num_states_y == 1:
        return (1, 0)
    _, x = np.unique(x, return_inverse=True)
    _, y = np.unique(y, return_inverse=True)
    
    x = x - min(x)
    y = y - min(y)

    n_mat = hist3(x, y, range(num_states_x), range(num_states_y))
    
    p = np.sum(n_mat, axis=1)  # ?
    w = np.sum(n_mat, axis=0)  # ?
    nullerp = len(p) - np.count_nonzero(p)  # ?
    nullerw = len(w) - np.count_nonzero(w)
    lengthX = len(x)
    T = 0
    for i in range(num_states_x):
        for j in range(num_states_y):
            if p[i] and w[j] != 0:
                n_star = (p[i] * w[j]+ 0.0) / (lengthX+0.0)
                T += (n_mat[i][j] - n_star + 0.0) ** 2 / n_star*1.0
    degrees = (num_states_x - 1 - nullerp) * (num_states_y - 1 - nullerw)
    if degrees == 0:
        degrees = 1
    result = 1 - chi2.cdf(T, degrees)
    return (result, T)

def chiSqQuant(x, y, num_states_x, num_states_y):
    if num_states_x == 1 or num_states_y == 1:
        return (1, 0)    
    x = x - min(x)
    y = y - min(y)
    n_mat = hist3(x, y, range(num_states_x), range(num_states_y))
    T, result, _, _ = chi2_contingency(n_mat)
    return (result, T)


def calculateEps(Y, yhat, cyclic):
    if cyclic:
        return np.remainder(Y - yhat, max(Y) - min(Y) + 1)
    return Y - yhat


def getSimplifiedArray(inputArray, num):
    _, bin_edges = np.histogram(inputArray, num)
    return np.array([bin_edges[element-1] for element in np.searchsorted(bin_edges, inputArray, side="right")])


def fitDiscrete(Xaux, Yaux, level, cyclic):
    # parameter
    num_iter = 8

    # rescaling:
    # X_new takes values from 1...X_new_max
    # Y_values are everything between Y_min and Y_max
    if len(np.unique(Xaux)) > 300:
        X = getSimplifiedArray(Xaux, 100)
    else:
        X = Xaux
    if len(np.unique(Yaux)) > 100:
        Y = getSimplifiedArray(Yaux, 100)
    else:
        Y = Yaux
    if cyclic:
        num_pos_fct = int(math.ceil(min(np.max(Y) - np.min(Y), 10)))
    else:
        num_pos_fct = int(math.ceil(min(np.max(Y) - np.min(Y), 20)))
    (X_values, aa, X_new) = np.unique(X, return_index=True, return_inverse=True)
    Y_values = np.array([i for i in range(int(math.floor(np.min(Y))), int(math.ceil(np.max(Y)))+1)])
    permElements = len(X_values)
    permElementsy = len(Y_values)
    
    if permElements == 1 or permElementsy == 1:
        fct = np.empty(permElements)
        fct.fill([Y_values[0]])
        p_val = 1
    else:
        p = hist3(X, Y, X_values, Y_values)  # ?
        fct = np.zeros(permElements)
        cand = np.zeros((permElements,permElementsy))
        i = 0
        for subp in p:
            argmax = np.argmax(subp)
            for k in range(argmax):
                subp[k] = subp[k] + 1.0 / (2.0 * (argmax-k))
            for k in range(argmax+1, len(subp)):
                subp[k] = subp[k] + 1.0 / (2.0 * (k-argmax))
            subp[argmax] = subp[argmax] + 1
            b = np.argsort(subp)
            cand[i] = b
            fct[i] = Y_values[b[-1]]
            i += 1
        yhat = fct[X_new]
        eps = calculateEps(Y, yhat, cyclic)
        if len(np.unique(eps)) == 1:
            p_val = 1
        else:
            measurement, eps = np.unique(eps, return_inverse=True)
            p_val = chiSqQuant(eps, X_new, len(measurement), permElements)
            p_val = p_val[0]
        i = 0
        if permElements > 30:
            pe = 30
        else:
            pe = permElements
        while p_val < level and i < num_iter:
            for j_new in np.random.permutation(range(pe)):
                pos_fct = np.zeros((num_pos_fct, len(fct)))
                p_val_comp = np.zeros(num_pos_fct)
                p_val_comp2 = np.zeros(num_pos_fct)
                candjnew = cand[j_new]
                lencandjnew = len(candjnew)
                for j in range(num_pos_fct):
                    pos_fct[j] = fct
                    pos_fct[j][j_new] = Y_values[candjnew[lencandjnew - j - 1]]
                    yhat = pos_fct[j][X_new]
                    eps = calculateEps(Y, yhat, cyclic)
                    measurement, eps = np.unique(eps, return_inverse=True)
                    (p_val_comp[j], p_val_comp2[j]) = chiSqQuant(eps, X_new, len(measurement), permElements)
                aa = np.max(p_val_comp)
                if aa < 1e-3:
                    j_max = np.argmin(p_val_comp2)
                else:
                    j_max = np.argmax(p_val_comp)
                fct = pos_fct[j_max]
                yhat = fct[X_new]
                eps = calculateEps(Y, yhat, cyclic)
                measurement, eps = np.unique(eps, return_inverse=True)
                p_val = chiSqQuant(eps, X_new, len(measurement), permElements)
                p_val = p_val[0]
            i = i + 1
        if not cyclic:
            fct = fct + round(np.mean(eps))
    return (fct, p_val)

def fitBothDirDiscrete(X, cycX, Y, cycY, level, fw):
    """
        fits a discrete additive noise model in both directions X->Y and Y->X.
        - X and Y should both be of size (n,1), 
        - cycX is True if X should be modelled as a cyclic variable and False if not
        - cycY is True if Y should be modelled as a cyclic variable and False if not
        - level denotes the significance level of the independent test after which the algorithm 
        should stop looking for a solution
        - example:
        pars.p_X=[0.1 0.3 0.1 0.1 0.2 0.1 0.1];pars.X_values=[-3;-2;-1;0;1;3;4];
        pars2.p_n=[0.2 0.5 0.3];pars2.n_values=[-1;0;1];
        [X Y]=add_noise(500,@(x) round(0.5*x.^2),'custom',pars,'custom',pars2, 'fct');
        
        [fct1 p_val1 fct2 p_val2]=fit_both_dir_discrete(X,0,Y,0,0.05,0);   
    """
    if fw:
        (_, p_val_fw) = fitDiscrete(X, Y, level, cycY)
        return p_val_fw
    else:
        (_, p_val_bw) = fitDiscrete(Y, X, level, cycX)
        return p_val_bw

