CausalityCompetition
====================

Causality competition in kaggle, position 34. https://www.kaggle.com/c/cause-effect-pairs

Requirements: 
Python 2.7 along with the following packages:

 - pandas (tested with version 10.1)
 - sklearn (tested with version 0.13)
 - numpy (tested with version 1.6.2)
 - scipy (tested with version 0.10.)

To run the program,

1. Download the data(https://www.kaggle.com/c/cause-effect-pairs/data)
2. Modify SETTINGS.json to point to the training and validation data on your system, as well as a place to save the trained model and a place to save the submission
3. Train the model by running `python CausalityTrainer.py`
4. Make predictions on the validation set by running `python CausalityPredictor.py`
