import numpy as np
from sklearn.base import BaseEstimator
from scipy.special import psi
from scipy.stats.stats import pearsonr
import pwling
import math
from fitBothDirDiscrete import fitBothDirDiscrete
from scipy.stats import skew, skewtest, kurtosis, kurtosistest, ks_2samp, kendalltau, linregress, normaltest
from sklearn.feature_extraction.text import CountVectorizer

class FeatureFunctions:
    def __init__(self):
        self.arrayMax = np.zeros(7831)
        self.arrayMean = np.zeros(7831)
        self.arrayBinary = np.zeros(7831)
        self.arrayMaxRev = np.zeros(7831)
        self.arrayMeanRev = np.zeros(7831)
        self.arrayBinaryRev = np.zeros(7831)
        self.boolMax = True
        self.boolMean = True
        self.boolBinary = True
        self.boolVar = True
        self.boolVarRev = True
        self.boolMaxRev = True
        self.boolMeanRev = True
        self.boolBinaryRev = True
        self.counter = 0
            
    def ks(self, x,y):
        return ks_2samp(x, y)[1]
    
    def kdt(self, x, y):
        return kendalltau(x, y)[1]
    
    def lr(self, x, y):
        return linregress(x, y)[3]
    
    def nt(self, x):
        return normaltest(x)[1]
    
    def identity(self, x):
        return x

    def count_unique(self, x):
        return len(set(x))
    
    def autocorr(self, x):
        return len(np.correlate(x, x, mode='full'))
    
    def mean_autocorr(self, x):
        return (1.0-(self.autocorr(x)+0.0)/(2*len(x)+0.0))*100.0
    
    def normalized_entropy(self, x):
        x = (x - np.mean(x)) / np.std(x)
        x = np.sort(x)
        hx = 0.0;
        for i in range(len(x)-1):
            delta = x[i+1] - x[i];
            if delta != 0:
                hx += np.log(np.abs(delta));
        hx = hx / (len(x) - 1) + psi(len(x)) - psi(1);
        return hx
    
    def divideInBuckets(self, x, valuesArray, n):
        division = (len(valuesArray) +0.0)/(0.0 + float(n))
        return [ valuesArray[int(round(division * i)): int(round(division * (i + 1)))] for i in xrange(n) ]
    
    def vertical_line_test_max(self, x, y):
        if self.boolMax:
            self.boolMax = False
            self.counter = 0
        self.counter = self.counter + 1
        return self.arrayMax[self.counter-1]
    
    def vertical_line_test_mean(self, x, y):
        if self.boolMean:
            self.boolMean = False
            self.counter = 0
        self.counter = self.counter + 1
        return self.arrayMean[self.counter-1]

    def fbdd(self, x, y, atype, btype):
        result =  fitBothDirDiscrete(x, False, y, False, 0.05, True)
        print result
        return result

    def fbdd_rev(self, x, y, atype, btype):
        result = fitBothDirDiscrete(x, False, y, False, 0.05, False)
        print result
        return result
    
    def vertical_line_test_var(self, x, yaux):
        if self.boolVar:
            self.boolVar = False
            self.counter = 0
        if yaux.max() - yaux.min() == 0:
            return 0
        y = (yaux + 0.0 - yaux.min())/(yaux.max() + 0.0 -yaux.min())
        uniqueValues = np.unique(x)
        varianceFinal = []
        if len(uniqueValues) > 500:
            uniqueValuesPrima = self.divideInBuckets(x, uniqueValues, 100)
            for bucket in uniqueValuesPrima:
                generalPositions = []
                for element in bucket:
                    positions = np.where(x == element)[0]
                    generalPositions.extend(positions)
                if len(generalPositions) == 1:
                    varianceFinal.append(0)
                    continue
                variance = y[generalPositions].var(ddof=1)
                if not math.isnan(variance):
                    varianceFinal.append(variance)      
        else:
            for element in uniqueValues:
                positions = np.where(x == element)[0]
                if len(positions) == 1:
                    varianceFinal.append(0)
                    continue
                variance = y[positions].var(ddof=1)
                if not math.isnan(variance):
                    varianceFinal.append(variance)
        if len(varianceFinal) == 0 or len(varianceFinal) == 1:
            return 0
        vf = np.array(varianceFinal)
        self.arrayMean[self.counter] = vf.mean()
        maxResult = np.max(vf)
        self.arrayMax[self.counter] = maxResult
        if maxResult != 0:
            self.arrayBinary[self.counter] = 1
        else:
            self.arrayBinary[self.counter] = 0
        self.counter = self.counter + 1
        result = vf.var()
        if not math.isnan(result):
            return result
        return 0
    
    def vertical_line_test_var_rev(self, yaux, x):
        if self.boolVarRev:
            self.boolVarRev = False
            self.counter = 0
        uniqueValues = np.unique(x)
        if yaux.max() - yaux.min() == 0:
            return 0
        y = (yaux + 0.0 - yaux.min())/(yaux.max() + 0.0 -yaux.min())
        varianceFinal = []
        if len(uniqueValues) > 500:
            uniqueValuesPrima = self.divideInBuckets(x, uniqueValues, 100)
            for bucket in uniqueValuesPrima:
                generalPositions = []
                for element in bucket:
                    positions = np.where(x == element)[0]
                    generalPositions.extend(positions)
                if len(generalPositions) == 1:
                    varianceFinal.append(0)
                    continue
                variance = y[generalPositions].var(ddof=1)
                if not math.isnan(variance):
                    varianceFinal.append(variance)      
        else:
            for element in uniqueValues:
                positions = np.where(x == element)[0]
                if len(positions) == 1:
                    varianceFinal.append(0)
                    continue
                variance = y[positions].var(ddof=1)
                if not math.isnan(variance):
                    varianceFinal.append(variance)
        if len(varianceFinal) == 0 or len(varianceFinal) == 1:
            return 0
        vf = np.array(varianceFinal)
        self.arrayMeanRev[self.counter] = vf.mean()
        maxResult = np.max(vf)
        self.arrayMaxRev[self.counter] = maxResult
        if maxResult != 0:
            self.arrayBinaryRev[self.counter] = 1
        else:
            self.arrayBinaryRev[self.counter] = 0
        self.counter = self.counter + 1
        result = vf.var()
        if not math.isnan(result):
            return result
        return 0
    
    def vertical_line_test_binary(self, x, y):
        if self.boolBinary:
            self.boolBinary = False
            self.counter = 0
        self.counter = self.counter + 1
        return self.arrayBinary[self.counter-1]
    
    def vertical_line_test_max_rev(self, y, x):
        if self.boolMaxRev:
            self.boolMaxRev = False
            self.counter = 0
        self.counter = self.counter + 1
        return self.arrayMaxRev[self.counter-1]
    
    def vertical_line_test_mean_rev(self, y, x):
        if self.boolMeanRev:
            self.boolMeanRev = False
            self.counter = 0
        self.counter = self.counter + 1
        return self.arrayMeanRev[self.counter-1]
    
    def skewness(self, x):
        return skewtest(x)[0]
    
    def skewnessp(self, x):
        return skewtest(x)[1]
    
    def kurtosist(self, x):
        return kurtosistest(x)[0]

    def kurtosistp(self, x):
        return kurtosistest(x)[1]
    
    def kurt(self, x):
        return kurtosis(x)
    
    def vertical_line_test_binary_rev(self, y, x):
        if self.boolBinaryRev:
            self.boolBinaryRev = False
            self.counter = 0
        self.counter = self.counter + 1
        return self.arrayBinaryRev[self.counter-1]
    
    def entropy_difference(self, x, y):
        return self.normalized_entropy(x) - self.normalized_entropy(y)
    
    def correlation(self, x, y):
        return pearsonr(x, y)[0]
    
    def correlation_magnitude(self, x, y):
        return abs(self.correlation(x, y))
    
    def pwling1(self, x, y):
        arrayToTest  = np.array([x,y])
        return pwling.doPwling(arrayToTest, 1)
    
    def pwling2(self, x, y):
        arrayToTest  = np.array([x,y])
        return pwling.doPwling(arrayToTest, 2)
    
    def pwling3(self, x, y):
        arrayToTest  = np.array([x,y])
        return pwling.doPwling(arrayToTest, 3)
    
    def pwling4(self, x, y):
        arrayToTest  = np.array([x,y])
        return pwling.doPwling(arrayToTest, 4)
    
    def pwling5(self, x, y):
        arrayToTest  = np.array([x,y])
        return pwling.doPwling(arrayToTest, 5)

ff = FeatureFunctions()

class FeatureMapper:
    def __init__(self, features):
        self.features = features

    def fit(self, X, y=None):
        for _, column_names, extractor in self.features:
            extractor.fit(X[column_names], y)

    def transform(self, X):
        extracted = []
        for feature_name, column_names, extractor in self.features:
            print feature_name
            fea = extractor.transform(X[column_names])
            
            if hasattr(fea, "toarray"):
                extracted.append(fea.toarray())
            else:
                extracted.append(fea)
        if len(extracted) > 1:
            return np.concatenate(extracted, axis=1)
        else: 
            return extracted[0]

    def fit_transform(self, X, y=None):
        extracted = []
        for feature_name, column_names, extractor in self.features:
            print feature_name
            fea = extractor.fit_transform(X[column_names], y)
            if hasattr(fea, "toarray"):
                extracted.append(fea.toarray())
            else:
                extracted.append(fea)
        print "Concatenating...."
        if len(extracted) > 1:
            return np.concatenate(extracted, axis=1)
        else: 
            return extracted[0]

class SimpleTransform(BaseEstimator):
    def __init__(self, transformer=ff.identity):
        self.transformer = transformer

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def transform(self, X, y=None):
        return np.array([self.transformer(x) for x in X], ndmin=2).T

class MultiColumnTransform(BaseEstimator):
    def __init__(self, transformer):
        self.transformer = transformer

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None):
        return self.transform(X)
    
    def transform(self, X, y=None):
        return np.array([self.transformer(*x[1]) for x in X.iterrows()], ndmin=2).T
    
    
preprocessedFeatures = ["A: Skewness", "A: Skewtest", "B: Skewness", "B: Skewtest", "A: Skewtest pvalue", 
                        'A: Kurtosis', 'A: Kurtosistest', 'B: normaltest pvalue', 'B: Skewtest', 
                        'B: Skewness', 'A: normaltest pvalue', 'A: Kurtosistest pvalue', "A: Autocorrelation", 
                        "A: Mean autocorrelation", "B: Autocorrelation", "B: Mean autocorrelation",
                        'B: Kurtosistest pvalue', 'B: Kurtosistest', 'B: Kurtosis', 'B: Skewtest pvalue', 
                        'Kendall Tau', 'Linear regression pvalue', 'Ks 2samp', "VLTVar", "VLTMax", "VLTMean", 
                        "VLTBinary", "VLTVarReversed", "VLTMaxReversed","VLTMeanReversed", "VLTBinaryReversed", 
                        "Pwling1", "Pwling2", "Pwling3", "Pwling4", "Pwling5", "A: Normalized Entropy", 
                        "B: Normalized Entropy", "Entropy Difference" ]

features = [['Number of Samples', 'A', SimpleTransform(transformer=len)],
                ['Type of Samples A', 'A type', CountVectorizer()],
                ['Type of Samples B', 'B type', CountVectorizer()],
                ['A: Skewness', 'A', SimpleTransform(transformer=skew)],
                ['A: Skewtest', 'A', SimpleTransform(transformer=ff.skewness)],
                ['A: Skewtest pvalue', 'A', SimpleTransform(transformer=ff.skewnessp)],
                ['A: Kurtosis', 'A', SimpleTransform(transformer=ff.kurt)],
                ['A: Kurtosistest', 'A', SimpleTransform(transformer=ff.kurtosist)],
                ['A: Kurtosistest pvalue', 'A', SimpleTransform(transformer=ff.kurtosistp)],
                ['A: normaltest pvalue', 'A', SimpleTransform(transformer=ff.nt)],
                ['A: Mean autocorrelation', 'A', SimpleTransform(transformer=ff.mean_autocorr)],
                ['A: Number of Unique Samples', 'A', SimpleTransform(transformer=ff.count_unique)],
                ['B: Number of Unique Samples', 'B', SimpleTransform(transformer=ff.count_unique)],
                ['B: Skewness', 'B', SimpleTransform(transformer=ff.skew)],
                ['B: Skewtest', 'B', SimpleTransform(transformer=ff.skewness)],
                ['B: normaltest pvalue', 'B', SimpleTransform(transformer=ff.nt)],
                ['Ks 2samp', ['A','B'], MultiColumnTransform(ff.ks)],
                ['Linear regression pvalue', ['A','B'], MultiColumnTransform(ff.lr)],
                ['Kendall Tau', ['A','B'], MultiColumnTransform(ff.kdt)],
                ['B: Skewtest pvalue', 'B', SimpleTransform(transformer=ff.skewnessp)],
                ['B: Kurtosis', 'B', SimpleTransform(transformer=ff.kurt)],
                ['B: Kurtosistest', 'B', SimpleTransform(transformer=ff.kurtosist)],
                ['B: Kurtosistest pvalue', 'B', SimpleTransform(transformer=ff.kurtosistp)],
                ['VLTVar', ['A','B'], MultiColumnTransform(ff.vertical_line_test_var)],
                ['VLTMax', ['A','B'], MultiColumnTransform(ff.vertical_line_test_max)],
                ['VLTMean', ['A','B'], MultiColumnTransform(ff.vertical_line_test_mean)],
                ['VLTVarReversed', ['A','B'], MultiColumnTransform(ff.vertical_line_test_var_rev)],
                ['VLTMaxReversed', ['A','B'], MultiColumnTransform(ff.vertical_line_test_max_rev)],
                ['VLTMeanReversed', ['A','B'], MultiColumnTransform(ff.vertical_line_test_mean_rev)],
                ['A: Normalized Entropy', 'A', SimpleTransform(transformer=ff.normalized_entropy)],
                ['B: Normalized Entropy', 'B', SimpleTransform(transformer=ff.normalized_entropy)],
                ['Pwling1', ['A','B'], MultiColumnTransform(ff.pwling1)],
                ['Pwling2', ['A','B'], MultiColumnTransform(ff.pwling2)],
                ['Pwling3', ['A','B'], MultiColumnTransform(ff.pwling3)],
                ['Pwling4', ['A','B'], MultiColumnTransform(ff.pwling4)],
                ['Pwling5', ['A','B'], MultiColumnTransform(ff.pwling5)],
                ['Pearson R', ['A','B'], MultiColumnTransform(ff.correlation)],
                ['Pearson R Magnitude', ['A','B'], MultiColumnTransform(ff.correlation_magnitude)],
                ['Entropy Difference', ['A','B'], MultiColumnTransform(ff.entropy_difference)]]