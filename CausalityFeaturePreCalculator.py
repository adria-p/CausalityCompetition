import data_io
import CausalityFeatureFunctions as f

class FeaturePreCalculator:
    def __init__(self, getTrain=True):
        self.getTrain = getTrain
        """
            Features to ignore when calculating. Like, for example: 
            
            ["A: Autocorrelation", "B: Autocorrelation",  "A: Mean autocorrelation", "B: Mean autocorrelation", 
            "VLTVar", "VLTMax", "VLTMean", "VLTBinary", "VLTVarReversed", "VLTMaxReversed","VLTMeanReversed", 
            "VLTBinaryReversed", "Pwling1", "Pwling2", "Pwling3", "Pwling4", "Pwling5", "A: Normalized Entropy", 
            "B: Normalized Entropy", "Entropy Difference" ]
            
        """
        self.ignoreFeatures = []
        self.dataToWrite = [['A: Autocorrelation', 'A', f.SimpleTransform(transformer=f.ff.autocorr)],
            ['A: Skewness', 'A', f.SimpleTransform(transformer=f.ff.skew)],
            ['A: Skewtest', 'A', f.SimpleTransform(transformer=f.ff.skewness)],
            ['A: Skewtest pvalue', 'A', f.SimpleTransform(transformer=f.ff.skewnessp)],
            ['A: Kurtosis', 'A', f.SimpleTransform(transformer=f.ff.kurt)],
            ['A: Kurtosistest', 'A', f.SimpleTransform(transformer=f.ff.kurtosist)],
            ['A: Kurtosistest pvalue', 'A', f.SimpleTransform(transformer=f.ff.kurtosistp)],
            ['A: normaltest pvalue', 'A', f.SimpleTransform(transformer=f.ff.nt)],
            ['A: Mean autocorrelation', 'A', f.SimpleTransform(transformer=f.ff.mean_autocorr)],
            ['B: Autocorrelation', 'B', f.SimpleTransform(transformer=f.ff.autocorr)],
            ['B: Mean autocorrelation', 'B', f.SimpleTransform(transformer=f.ff.mean_autocorr)],
            ['B: Skewness', 'B', f.SimpleTransform(transformer=f.ff.skew)],
            ['B: Skewtest', 'B', f.SimpleTransform(transformer=f.ff.skewness)],
            ['B: normaltest pvalue', 'B', f.SimpleTransform(transformer=f.ff.nt)],
            ['Ks 2samp', ['A','B'], f.MultiColumnTransform(f.ff.ks)],
            ['Linear regression pvalue', ['A','B'], f.MultiColumnTransform(f.ff.lr)],
            ['Kendall Tau', ['A','B'], f.MultiColumnTransform(f.ff.kdt)],
            ['B: Skewtest pvalue', 'B', f.SimpleTransform(transformer=f.ff.skewnessp)],
            ['B: Kurtosis', 'B', f.SimpleTransform(transformer=f.ff.kurt)],
            ['B: Kurtosistest', 'B', f.SimpleTransform(transformer=f.ff.kurtosist)],
            ['B: Kurtosistest pvalue', 'B', f.SimpleTransform(transformer=f.ff.kurtosistp)],
            ['VLTVar', ['A','B'], f.MultiColumnTransform(f.ff.vertical_line_test_var)],
            ['VLTMax', ['A','B'], f.MultiColumnTransform(f.ff.vertical_line_test_max)],
            ['VLTMean', ['A','B'], f.MultiColumnTransform(f.ff.vertical_line_test_mean)],
            ['VLTBinary', ['A','B'], f.MultiColumnTransform(f.ff.vertical_line_test_binary)],
            ['VLTVarReversed', ['A','B'], f.MultiColumnTransform(f.ff.vertical_line_test_var_rev)],
            ['VLTMaxReversed', ['A','B'], f.MultiColumnTransform(f.ff.vertical_line_test_max_rev)],
            ['VLTMeanReversed', ['A','B'], f.MultiColumnTransform(f.ff.vertical_line_test_mean_rev)],
            ['VLTBinaryReversed', ['A','B'], f.MultiColumnTransform(f.ff.vertical_line_test_binary_rev)],
            ['Pwling1', ['A','B'], f.MultiColumnTransform(f.ff.pwling1)],
            ['Pwling2', ['A','B'], f.MultiColumnTransform(f.ff.pwling2)],
            ['Pwling3', ['A','B'], f.MultiColumnTransform(f.ff.pwling3)],
            ['Pwling4', ['A','B'], f.MultiColumnTransform(f.ff.pwling4)],
            ['Pwling5', ['A','B'], f.MultiColumnTransform(f.ff.pwling5)],
            ['A: Normalized Entropy', 'A', f.SimpleTransform(transformer=f.ff.normalized_entropy)],
            ['B: Normalized Entropy', 'B', f.SimpleTransform(transformer=f.ff.normalized_entropy)],
            ['Entropy Difference', ['A','B'], f.MultiColumnTransform(f.ff.entropy_difference)]]

    def getDataset(self):
        if self.getTrain:
            readData = data_io.read_train_pairs()
            readData2 = data_io.read_train_info()
        else:
            readData = data_io.read_valid_pairs()
            readData2 = data_io.read_valid_info()
        readData["A type"] = readData2["A type"]
        readData["B type"] = readData2["B type"]
        return readData

    def run(self):
        print "Reading in the data"
        dataset = self.getDataset()    
        featureNames = [i[0] for i in self.dataToWrite]
        if self.ignoreFeatures != []:
            if self.getTrain:
                intermediate = data_io.read_intermediate_train()
            else:
                intermediate = data_io.read_intermediate_valid()
            for i in self.ignoreFeatures:
                dataset[i] = intermediate[i]
        for element in self.dataToWrite:
            if element[0] in self.ignoreFeatures:
                element[1] = element[0]
                element[2] = f.SimpleTransform(transformer=f.ff.identity)
        print "Extracting features and transforming"
        featureMapper = f.FeatureMapper(self.dataToWrite)
        transformedDataset = featureMapper.transform(dataset)
        print "Saving the data"
        if self.getTrain:
            data_io.write_intermediate_train(featureNames, transformedDataset, dataset)
        else:
            data_io.write_intermediate_valid(featureNames, transformedDataset, dataset)
    
if __name__=="__main__":
    fpc = FeaturePreCalculator()
    fpc.run()