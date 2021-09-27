import os
from math import sqrt

import numpy as np
import pandas
from numpy import full
from scipy.io import arff
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import classification_report, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB, BernoulliNB

from ud_bagging import UDBaggingClassifier


def read_data(file):
    with open(file) as f:
        data, meta = arff.loadarff(f)
    data_frame = pandas.DataFrame(data)
    for col in data_frame:
        if col == 'entrez':
            data_frame[col] = list(map(lambda x: int(x), data_frame[col]))
        elif col == 'class':
            data_frame[col] = list(map(lambda x: int(x.decode('UTF-8')), data_frame[col]))
        else:
            data_frame[col] = list(map(lambda x: float(x), data_frame[col]))
    return data_frame


def cross_validation(model, data_frame, n_estimators, random_seed):
    first_feat = 1

    uncertain_features = full((data_frame.shape[1] - (first_feat + 1),), False)
    for col in data_frame:
        if col != 'entrez' and col != 'class':
            uncertain_features[data_frame.columns.get_loc(col) - first_feat] = True

    precision_anti = []
    recall_anti = []
    f1score_anti = []
    support_anti = []
    precision_pro = []
    recall_pro = []
    f1score_pro = []
    support_pro = []
    accuracy = []
    balanced_accuracy = []
    roc_auc = []
    g_mean = []

    kf = KFold(n_splits=10, shuffle=True, random_state=random_seed)
    i = 0
    for trainIndices, testIndices in kf.split(data_frame):
        train_x = data_frame.iloc[trainIndices, first_feat:-1]
        train_y = data_frame.iloc[trainIndices].loc[:, 'class']

        test_x = data_frame.iloc[testIndices, first_feat:-1]
        test_y = data_frame.iloc[testIndices].loc[:, 'class']

        if model == 'NB-NV':
            classifier = GaussianNB()
        elif model == 'NB-EV':
            classifier = BernoulliNB(binarize=0.5)
        elif model == 'ENB-NV':
            classifier = BaggingClassifier(base_estimator=GaussianNB(), n_estimators=n_estimators,
                                           max_features=max(1, int(np.sqrt(train_x.shape[1]))), n_jobs=4,
                                           random_state=random_seed)
        elif model == 'ENB-EV':
            classifier = BaggingClassifier(base_estimator=BernoulliNB(binarize=0.5),
                                           n_estimators=n_estimators,
                                           max_features=max(1, int(np.sqrt(train_x.shape[1]))), n_jobs=4,
                                           random_state=random_seed)
        elif model == 'ENB-NV+BRS':
            classifier = UDBaggingClassifier(base_estimator=GaussianNB(), n_estimators=n_estimators,
                                             max_features=max(1, int(np.sqrt(train_x.shape[1]))), n_jobs=4,
                                             random_state=random_seed, uncertain_features=uncertain_features,
                                             biased_subspaces=True)
        else:   # model == 'ENB-EV+BRS'
            classifier = UDBaggingClassifier(base_estimator=BernoulliNB(binarize=0.5),
                                             n_estimators=n_estimators,
                                             max_features=max(1, int(np.sqrt(train_x.shape[1]))), n_jobs=4,
                                             random_state=random_seed, uncertain_features=uncertain_features,
                                             biased_subspaces=True)

        classifier.fit(train_x, train_y)

        proba = classifier.predict_proba(test_x)
        pred_y = classifier.classes_.take(np.argmax(proba, axis=1), axis=0)

        report = classification_report(test_y, pred_y, target_names=['anti', 'pro'], output_dict=True,
                                       zero_division=0)

        precision_anti.append(report['anti']['precision'])
        recall_anti.append(report['anti']['recall'])
        f1score_anti.append(report['anti']['f1-score'])
        support_anti.append(report['anti']['support'])
        precision_pro.append(report['pro']['precision'])
        recall_pro.append(report['pro']['recall'])
        f1score_pro.append(report['pro']['f1-score'])
        support_pro.append(report['pro']['support'])
        accuracy.append(report['accuracy'])
        balanced_accuracy.append(balanced_accuracy_score(test_y, pred_y))
        roc_auc.append(roc_auc_score(test_y, proba[:, 1]))
        g_mean.append(sqrt(report['anti']['recall'] * report['pro']['recall']))

        i += 1

    results = pandas.DataFrame(data={'precision(anti)': precision_anti, 'recall(anti)': recall_anti,
                                     'f1-score(anti)': f1score_anti, 'support(anti)': support_anti,
                                     'precision(pro)': precision_pro, 'recall(pro)': recall_pro,
                                     'f1-score(pro)': f1score_pro, 'support(pro)': support_pro,
                                     'accuracy': accuracy, 'b-accuracy': balanced_accuracy,
                                     'roc-auc': roc_auc, 'g-mean': g_mean})
    results.loc['mean'] = results.mean()
    results['f1-score(anti)']['mean'] = 2 * results['precision(anti)']['mean'] * results['recall(anti)']['mean']
    if results['f1-score(anti)']['mean'] != 0:
        results['f1-score(anti)']['mean'] /= results['precision(anti)']['mean'] + results['recall(anti)']['mean']
    results['f1-score(pro)']['mean'] = 2 * results['precision(pro)']['mean'] * results['recall(pro)']['mean']
    if results['f1-score(pro)']['mean'] != 0:
        results['f1-score(pro)']['mean'] /= results['precision(pro)']['mean'] + results['recall(pro)']['mean']
    results['f1-score(pro)']['std'] = np.nan
    results['g-mean']['mean'] = sqrt(results['recall(anti)']['mean'] * results['recall(pro)']['mean'])
    print(results)


models = ['NB-NV', 'ENB-NV', 'ENB-NV+BRS', 'NB-EV', 'ENB-EV', 'ENB-EV+BRS']

pandas.set_option('display.max_rows', None, 'display.max_columns', None, 'display.width', 170)
datasets_dir = '../data/'
for root, directories, files in os.walk(datasets_dir):
    datasets = files
    break
estimators = 500
seed = 2

for dataset in datasets:
    df = read_data(datasets_dir + dataset)
    for model in models:
        print()
        print('dataset =', dataset, ' | model =', model)
        cross_validation(model, df, estimators, seed)
