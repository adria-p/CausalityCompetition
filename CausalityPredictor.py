import data_io
import CausalityFeatureFunctions as f

class CausalityPredictor:
    def getValidationDataset(self):
        print "Reading the valid pairs"
        valid = data_io.read_valid_pairs()
        valid2 = data_io.read_valid_info()
        valid["A type"] = valid2["A type"]
        valid["B type"] = valid2["B type"]
        return valid

    def run(self):
        valid = self.getValidationDataset()
        if f.preprocessedFeatures != []:
            intermediate = data_io.read_intermediate_valid()
            for i in f.preprocessedFeatures:
                valid[i] = intermediate[i]
        print "Loading the classifier"
        classifier = data_io.load_model()
        print "Making predictions"
        predictions = classifier.predict(valid)
        predictions = predictions.flatten()
        print "Writing predictions to file"
        data_io.write_submission(predictions)

if __name__=="__main__":
    cp = CausalityPredictor()
    cp.run()