import data_io
import CausalityFeatureFunctions as f
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

class CausalityTrainer:
    def __init__(self, directionForward=True):
        self.directionForward = directionForward

    def getFeatureExtractor(self, features):
        combined = f.FeatureMapper(features)
        return combined
    
    def getPipeline(self, feat):
        features = self.getFeatureExtractor(feat)
        steps = [("extract_features", features),
                 ("classify", RandomForestRegressor(compute_importances=True, n_estimators=500, 
                                                    verbose=2, n_jobs=1, min_samples_split=10, 
                                                    random_state=0))]
        return Pipeline(steps)
    
    def getTrainingDataset(self):
        print "Reading in the training data"
        train = data_io.read_train_pairs()
        print "Reading the information about the training data"
        train2 = data_io.read_train_info()
        train["A type"] = train2["A type"]
        train["B type"] = train2["B type"]
        return train
    
    def run(self):
        features = f.features
        train = self.getTrainingDataset()
        print "Reading preprocessed features"
        if f.preprocessedFeatures != []:
            intermediate = data_io.read_intermediate_train()
            for i in f.preprocessedFeatures:
                train[i] = intermediate[i]
            for i in features:
                if i[0] in f.preprocessedFeatures:
                    i[1] = i[0]
                    i[2] = f.SimpleTransform(transformer=f.ff.identity)
        print "Reading targets"
        target = data_io.read_train_target()
        print "Extracting features and training model"
        classifier = self.getPipeline(features)
        if self.directionForward:
            finalTarget = [ x*(x+1)/2 for x in target.Target]
        else:
            finalTarget = [ -x*(x-1)/2 for x in target.Target]
        classifier.fit(train, finalTarget)
        print classifier.steps[-1][1].feature_importances_
        print "Saving the classifier"
        data_io.save_model(classifier)
    
if __name__=="__main__":
    ct = CausalityTrainer()