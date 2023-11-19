import os
import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge
from sklearn import svm
from xgboost import XGBRegressor
import json
from omegaconf import OmegaConf, DictConfig
from typing import Any, Dict, Tuple, Type, cast

env_vars = ['bio_1', 'bio_2', 'bio_3', 'bio_4', 'bio_5',
            'bio_6', 'bio_7', 'bio_8', 'bio_9', 'bio_10', 'bio_11', 'bio_12',
            'bio_13', 'bio_14', 'bio_15', 'bio_16', 'bio_17', 'bio_18', 'bio_19',
            'bdticm', 'bldfie', 'cecsol', 'clyppt', 'orcdrc', 'phihox', 'sltppt',
            'sndppt']

#for kenya
#env_vars = ['bio_1', 'bio_2', 'bio_3', 'bio_4', 'bio_5',
#            'bio_6', 'bio_7', 'bio_8', 'bio_9', 'bio_10', 'bio_11', 'bio_12',
 #           'bio_13', 'bio_14', 'bio_15', 'bio_16', 'bio_17', 'bio_18', 'bio_19']

def set_up_omegaconf() -> DictConfig:
    """Helps with loading config files"""

    conf = OmegaConf.load("./configs/env_baseline_configs/env_baseline.yaml")
    command_line_conf = OmegaConf.from_cli()
    
    if "config_file" in command_line_conf:

        config_fn = command_line_conf.config_file

        if os.path.isfile(config_fn):
            user_conf = OmegaConf.load(config_fn)
            conf = OmegaConf.merge(conf, user_conf)
        else:
            raise FileNotFoundError(f"config_file={config_fn} is not a valid file")

    conf = OmegaConf.merge(
        conf, command_line_conf
    )
    # conf = set_data_paths(conf)
    conf = cast(DictConfig, conf)  # convince mypy that everything is alright

    return conf

def topkk(y, pred, numk=None):
    topk = []
    ks = []
    for i in range(len(y)):
        non_zeros = np.where(y[i] != 0)[0]
        k = len(non_zeros)
        if not numk is None:
            k = min(k,numk)
        if k==0:
            pass
        else:
            non_zeros = np.where(y[i] != 0)[0]

            non_zeros = np.argsort(y[i])[-k:]
            #ks +=[k]
            species_k = np.argsort(pred[i,:])[-k:]
            acc = len([j for j in species_k if j in non_zeros])
            acc = acc/k
            topk += [acc]
    return(np.mean(topk))

def get_topks(y, pred):
    
    top10 = topkk(y, pred, numk=10)
    top30 = topkk(y, pred, numk=30)
    topk = topkk(y, pred)
    return({"top10":top10, "top30":top30, "topk":topk})

def process(opts):
    if opts.subset != "":
        indices = np.load(opts.subset)
    else:
        indices = [i for i in range(opts.species)]
    num_species = len(indices)

    #hs = pd.read_csv(opts.hs_data)
    train = pd.read_csv(opts.train)
    val = pd.read_csv(opts.val)
    test = pd.read_csv(opts.test)
    #train = training.merge(hs, how="inner", left_on="hotspot", right_on="hotspot_id")
    #train = train.drop(columns=['Unnamed: 0_x', 'Unnamed: 0_y', 'r', 'g', 'b', 'nir'])
    #val = valid.merge(hs, how="inner", left_on="hotspot", right_on="hotspot_id")
    #val = val.drop(columns=['Unnamed: 0_x', 'Unnamed: 0_y', 'r', 'g', 'b', 'nir'])

    for c in env_vars:
        train[c].fillna(train[c].mean(), inplace=True)
        val[c].fillna(train[c].mean(), inplace=True)
        test[c].fillna(train[c].mean(), inplace = True)
    y_val = np.zeros((len(val), num_species))
    y_train = np.zeros((len(train), num_species))
    y_test = np.zeros((len(test), num_species))

    for i in range(len(train)):
        with open(os.path.join(opts.root_targets, train.iloc[i]["hotspot_id"]+".json"), "r") as s:
            spe = json.load(s)
        y_train[i, :] = np.array(spe["probs"])[indices]
    for i in range(len(val)):
        with open(os.path.join(opts.root_targets, val.iloc[i]["hotspot_id"]+".json"), "r") as s:
            spe = json.load(s)
        y_val[i, :] = np.array(spe["probs"])[indices]
    for i in range(len(test)):
        with open(os.path.join(opts.root_targets, test.iloc[i]["hotspot_id"]+".json"), "r") as s:
            spe = json.load(s)
        y_test[i, :] = np.array(spe["probs"])[indices]

    X_train = train[env_vars].to_numpy()
    X_val = val[env_vars].to_numpy()
    X_test = test[env_vars].to_numpy()
    print(X_train.shape)
    if opts.save_hs != "":
        np.save(os.path.join(opts.save_hs, "X_train.npy"), X_train)
        np.save(os.path.join(opts.save_hs, "X_val.npy"), X_val)
        np.save(os.path.join(opts.save_hs, "X_test.npy"), X_test)
        np.save(os.path.join(opts.save_hs, "y_train.npy"), y_train)
        np.save(os.path.join(opts.save_hs, "y_val.npy"), y_val)
        np.save(os.path.join(opts.save_hs, "y_test.npy"), y_test)

    return (X_train, X_val, X_test, y_train, y_val, y_test)


def train(opts):
    if opts.subset != "":
        indices = np.load(opts.subset)
    else:
        indices = [i for i in range(opts.species)]

    num_species = len(indices)
    print(num_species)
    if opts.process_hs:
        X_train, X_val, X_test, y_train, y_val, y_test = process(opts)

    else:
        y_train = np.load(opts.y_train)
        y_val = np.load(opts.y_val)
        y_test = np.load(opts.y_test)
        X_train = np.load(opts.X_train)
        X_val = np.load(opts.X_val)
        X_test = np.load(opts.X_test)
        print(X_train.shape)
        print(y_train.shape)

    if opts.predictor == "mean":
        means = y_train.mean(axis=0)
        means_val = np.tile(means, (y_val.shape[0], 1))
        means_test = np.tile(means, (y_test.shape[0], 1))
        if opts.path_means != "":
            np.save(opts.path_means, means)
        print("MSE val: ", (np.abs(y_val - means_val) ** 2).mean())
        print("MAE val:", np.abs(y_val - means_val).mean())
        print("MSE test: ", (np.abs(y_test - means_test) ** 2).mean())
        print("MAE test:", np.abs(y_test - means_test).mean())
        print("val", get_topks(y_val,  means_val))
        print("test", get_topks(y_test,  means_test))
    elif opts.predictor == "GBR":

        preds = []
        for rs in opts.random_state:
            print("using GBR")
            model = GradientBoostingRegressor(random_state=rs)

            print(X_train.shape, y_train.shape)
            clf = MultiOutputRegressor(model).fit(X_train, y_train)
            pred_val = clf.predict(X_val)
            pred_test = clf.predict(X_test)
            # preds += [pred]
            print("done fitting model " + str(rs))
            print("MSE val: ", (np.abs(y_val - pred_val) ** 2).mean())
            print("MAE val:", np.abs(y_val - pred_val).mean())
            print("MSE test: ", (np.abs(y_test - pred_test) ** 2).mean())
            print("MAE test:", np.abs(y_test - pred_test).mean())
            print("val", get_topks(y_val,  pred_val))
            print("test", get_topks(y_test,  pred_test))
        if opts.save_pred != "":
            np.save(os.path.join(opts.save_pred, f"pred_val_GBR_{rs}.npy"), pred_val)
            np.save(os.path.join(opts.save_pred, f"pred_test_GBR_{rs}.npy"), pred_test)
    elif opts.predictor == "XGBR":
        random_states = opts.random_state
        preds = []
        for rs in random_states:
            model = XGBRegressor(random_state=rs, n_estimators=50)
            clf = MultiOutputRegressor(model).fit(X_train, y_train)
            pred = clf.predict(X_val)
            preds += [pred]
            pred[pred < 0] = 0
            print("CE: ", np.nansum(- y_val * np.log(pred) - (1 - y_val) * np.log(1 - pred)))
            print("MSE : ", (np.abs(y_val - pred) ** 2).mean())
            print("MAE :", np.abs(y_val - pred).mean())
        if opts.save_pred != "":
            np.save(os.path.join(opts.save_pred), pred)

    else:
        if opts.predictor == "SVR":
            print("using SVR")
            model = svm.SVR(C=25)
        if opts.predictor == "Ridge":
            model = Ridge(random_state=0)
        clf = MultiOutputRegressor(model).fit(X_train, y_train)
        pred = clf.predict(X_val)
        print("MSE : ", (np.abs(y_val - pred) ** 2).mean())
        print("MAE :", np.abs(y_val - pred).mean())
        if opts.save_pred != "":
            np.save(os.path.join(opts.save_pred), pred)


if __name__ == "__main__":
    conf = set_up_omegaconf()
    train(conf)
    print("Done")
