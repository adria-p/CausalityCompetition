import csv
import json
import numpy as np
import os
import pandas as pd
import pickle

def get_paths():
    paths = json.loads(open("SETTINGS.json").read())
    for key in paths:
        paths[key] = os.path.expandvars(paths[key])
    return paths

def parse_dataframe(df):
    parse_cell = lambda cell: np.fromstring(cell, dtype=np.float, sep=" ")
    df = df.applymap(parse_cell)
    return df

def read_train_pairs():
    train_path = get_paths()["train_pairs_path"]
    return parse_dataframe(pd.read_csv(train_path, index_col="SampleID"))

def read_train_target():
    path = get_paths()["train_target_path"]
    df = pd.read_csv(path, index_col="SampleID")
    df = df.rename(columns = dict(zip(df.columns, ["Target", "Details"])))
    return df

def read_train_info():
    path = get_paths()["train_info_path"]
    return pd.read_csv(path, index_col="SampleID")

def read_valid_pairs():
    valid_path = get_paths()["valid_pairs_path"]
    return parse_dataframe(pd.read_csv(valid_path, index_col="SampleID"))

def read_valid_info():
    path = get_paths()["valid_info_path"]
    return pd.read_csv(path, index_col="SampleID")

def read_solution():
    solution_path = get_paths()["solution_path"]
    return pd.read_csv(solution_path, index_col="SampleID")

def save_model(model):
    out_path = get_paths()["model_path"]
    pickle.dump(model, open(out_path, "w"))

def load_model():
    in_path = get_paths()["model_path"]
    return pickle.load(open(in_path))

def read_submission():
    submission_path = get_paths()["submission_path"]
    return pd.read_csv(submission_path, index_col="SampleID")

def write_submission(predictions):
    submission_path = get_paths()["submission_path"]
    writer = csv.writer(open(submission_path, "w"), lineterminator="\n")
    valid = read_valid_pairs()
    rows = [x for x in zip(valid.index, predictions)]    
    writer.writerow(("SampleID", "Target"))
    writer.writerows(rows)
    
def write_mixed():
    sub1 = "basic_python_benchmark1.csv"
    sub2 = "basic_python_benchmark2.csv"
    array1 = pd.read_csv(sub1, index_col="SampleID")
    array2 = pd.read_csv(sub2, index_col="SampleID")
    predictions = []
    for i in range(len(array1["Target"])):
        predictions.append(array1["Target"][i]+array2["Target"][i])
    write_submission(predictions)

def write_intermediate_train(featureNames, result, ids):
    intermediate_path = get_paths()["intermediate_train_path"]
    writer = csv.writer(open(intermediate_path, "w"), lineterminator="\n")
    i = 0
    finalRows = []
    for index in ids.index:
        row = [index]
        for element in result[i]:
            row.append(element)
        finalRows.append(row)
        i = i+1
    names = ["SampleID"]
    for name in featureNames:
        names.append(name)
    writer.writerow(names)
    writer.writerows(finalRows)

def read_intermediate_train():
    train_path = get_paths()["intermediate_train_path"]
    return pd.read_csv(train_path, index_col="SampleID")

def write_intermediate_valid(featureNames, result, ids):
    intermediate_path = get_paths()["intermediate_valid_path"]
    writer = csv.writer(open(intermediate_path, "w"), lineterminator="\n")
    i = 0
    finalRows = []
    for index in ids.index:
        row = [index]
        for element in result[i]:
            row.append(element)
        finalRows.append(row)
        i = i+1
    names = ["SampleID"]
    for name in featureNames:
        names.append(name)
    writer.writerow(names)
    writer.writerows(finalRows)

def read_intermediate_valid():
    valid_path = get_paths()["intermediate_valid_path"]
    return pd.read_csv(valid_path, index_col="SampleID")

if __name__=="__main__":
    write_mixed()
