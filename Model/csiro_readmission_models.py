"""Independent CSIRO and historical readmission model experiments. No pooling.

Matches READMIT_model_comparison.ipynb model classes, parameter grids and
preprocessing. CSIRO splits/CV are patient-grouped; the historical split remains
the original stratified profile split. Default CSIRO tuning is train-only sampled.
"""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import argparse
import hashlib
import json
import os
import time
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split, GroupShuffleSplit, StratifiedGroupKFold, StratifiedKFold, GridSearchCV
from sklearn.metrics import (average_precision_score, roc_auc_score, brier_score_loss,
    accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score,
    precision_recall_curve, confusion_matrix)
from sklearn.inspection import permutation_importance

SEED = 42
H_NUM = ['ADMIT_LOS', 'ICU_DAYS', 'NUM_CHRONIC_COND', 'ORDER_TOTAL_CHARGES',
         'PATIENT_AGE', 'DRG_APR_SEVERITY', 'OPERATION_COUNT', 'NUM_VISITS', 'TOTAL_LOS']
H_CAT = ['DOCTOR', 'DEPARTMENT', 'DISCHARGED_TO', 'ORDER_SET_USED', 'GENDER',
         'Patient location', 'RACE_CD', 'DIAGNOSIS_ICD_GROUP', 'ICD9_TARGET',
         'DRG_APR_CODE', 'DIAGNOSIS_ICD_CODE', 'PROCEDURE_ICD_CODE', 'DX_CODE']
C_NUM = ['AGE_AT_ADMISSION', 'CSIRO_EPISODE_LOS_DAYS', 'PRIOR_INPATIENT_EPISODES', 'ACTIVE_CONDITION_CODES_AT_DISCHARGE']
C_CAT = ['GENDER']


def model_specs():
    return {
        'Logistic regression': (LogisticRegression(max_iter=3000, solver='liblinear', random_state=SEED),
            {'model__C': [.1, 1, 10], 'model__class_weight': [None, 'balanced']}),
        'Decision tree': (DecisionTreeClassifier(random_state=SEED, class_weight='balanced'),
            {'model__max_depth': [4, 8], 'model__min_samples_leaf': [10, 30]}),
        'Random forest': (RandomForestClassifier(n_estimators=150, n_jobs=1, random_state=SEED),
            {'model__max_features': ['sqrt', .7], 'model__min_samples_leaf': [5, 15]}),
        'Gradient boosting': (GradientBoostingClassifier(max_depth=2, min_samples_leaf=20, random_state=SEED),
            {'model__n_estimators': [100, 200], 'model__learning_rate': [.05, .1]}),
    }


def pipeline(estimator, numeric, categorical):
    prep = ColumnTransformer([
        ('numeric', Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())]), numeric),
        ('category', OneHotEncoder(handle_unknown='ignore', min_frequency=10, sparse_output=False), categorical),
    ], remainder='drop', sparse_threshold=0.0)
    return Pipeline([('preprocess', prep), ('model', estimator)])


def probability_scores(y, p):
    return dict(AP=average_precision_score(y, p), ROC_AUC=roc_auc_score(y, p), Brier=brier_score_loss(y, p))


def threshold_scores(y, p, threshold):
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(Threshold=threshold, Accuracy=accuracy_score(y, pred),
        Balanced_accuracy=balanced_accuracy_score(y, pred), Precision=precision_score(y, pred, zero_division=0),
        Recall=recall_score(y, pred, zero_division=0), F1=f1_score(y, pred, zero_division=0),
        TN=int(tn), FP=int(fp), FN=int(fn), TP=int(tp))


def f1_threshold(y, p):
    precision, recall, thresholds = precision_recall_curve(y, p)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[np.flatnonzero(np.isclose(f1, f1.max()))[-1]])


def grouped_split(X, y, groups):
    rest, test = next(GroupShuffleSplit(n_splits=1, test_size=.1, random_state=SEED).split(X, y, groups))
    train_local, val_local = next(GroupShuffleSplit(n_splits=1, test_size=1/9, random_state=SEED).split(X.iloc[rest], y.iloc[rest], groups.iloc[rest]))
    train, val = rest[train_local], rest[val_local]
    group_sets = [set(groups.iloc[i]) for i in [train, val, test]]
    assert not (group_sets[0] & group_sets[1] or group_sets[0] & group_sets[2] or group_sets[1] & group_sets[2])
    return train, val, test


def fit_dataset(name, X, y, groups, numeric, categorical, output, max_cv_patients=5000, jobs=2):
    started = time.time()
    if name == 'CSIRO':
        train, val, test = grouped_split(X, y, groups)
        rng = np.random.default_rng(SEED)
        ids = groups.iloc[train].unique()
        if max_cv_patients is not None and len(ids) > max_cv_patients:
            ids = rng.choice(ids, size=max_cv_patients, replace=False)
        tuning = train[groups.iloc[train].isin(ids).to_numpy()]
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
        cv_splits = list(cv.split(X.iloc[tuning], y.iloc[tuning], groups.iloc[tuning]))
        for a, b in cv_splits:
            assert not set(groups.iloc[tuning].iloc[a]) & set(groups.iloc[tuning].iloc[b])
    else:
        rest, test = train_test_split(np.arange(len(X)), test_size=.1, stratify=y, random_state=SEED)
        train, val = train_test_split(rest, test_size=1/9, stratify=y.iloc[rest], random_state=SEED)
        tuning = train
        cv_splits = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(X.iloc[tuning], y.iloc[tuning]))
    for a, b in cv_splits:
        assert y.iloc[tuning].iloc[a].nunique() == 2 and y.iloc[tuning].iloc[b].nunique() == 2
    split_rows, audit = [], []
    for label, idx in [('Train', train), ('Validation', val), ('Test', test)]:
        split_rows.append(dict(Dataset=name, Split=label, Rows=len(idx), Groups=groups.iloc[idx].nunique(),
            Positives=int(y.iloc[idx].sum()), Prevalence=float(y.iloc[idx].mean())))
        audit.append(pd.DataFrame({'row_index': X.index[idx], 'group_id': groups.iloc[idx].values, 'split': label}))
    pd.DataFrame(split_rows).to_csv(output / f'{name.lower()}_split_summary.csv', index=False)
    pd.concat(audit).to_csv(output / f'{name.lower()}_split_assignment.csv.gz', index=False, compression='gzip')
    rows, fitted = [], {}
    scoring = {'AP': 'average_precision', 'ROC_AUC': 'roc_auc', 'Brier': 'neg_brier_score'}
    for model_name, (estimator, grid) in model_specs().items():
        print(f'{name}: tuning {model_name}; {len(tuning):,} tuning rows / {len(train):,} full training rows', flush=True)
        search = GridSearchCV(pipeline(estimator, numeric, categorical), grid, cv=cv_splits, scoring=scoring,
            refit=False, n_jobs=jobs, error_score='raise')
        search.fit(X.iloc[tuning], y.iloc[tuning])
        best = int(np.argmax(search.cv_results_['mean_test_AP']))
        params = search.cv_results_['params'][best]
        model = pipeline(estimator, numeric, categorical).set_params(**params)
        model.fit(X.iloc[train], y.iloc[train])
        fitted[model_name] = model
        p_val = model.predict_proba(X.iloc[val])[:, 1]
        scores = probability_scores(y.iloc[val], p_val)
        rows.append(dict(Dataset=name, Model=model_name, CV_AP=search.cv_results_['mean_test_AP'][best],
            CV_AP_SD=search.cv_results_['std_test_AP'][best], CV_ROC_AUC=search.cv_results_['mean_test_ROC_AUC'][best],
            Validation_AP=scores['AP'], Validation_ROC_AUC=scores['ROC_AUC'], Validation_Brier=scores['Brier'],
            Validation_F1_threshold=f1_threshold(y.iloc[val], p_val), Best_params=json.dumps(params),
            Tuning_rows=len(tuning), Training_rows=len(train), Feature_count=len(numeric)+len(categorical)))
        pd.DataFrame(rows).to_csv(output / f'{name.lower()}_validation_progress.csv', index=False)
    dummy = DummyClassifier(strategy='prior').fit(X.iloc[train], y.iloc[train])
    fitted['Prior-probability baseline'] = dummy
    d = probability_scores(y.iloc[val], dummy.predict_proba(X.iloc[val])[:, 1])
    prior_cv_ap = np.array([y.iloc[tuning].iloc[b].mean() for _, b in cv_splits])
    rows.append(dict(Dataset=name, Model='Prior-probability baseline', CV_AP=float(prior_cv_ap.mean()),
        CV_AP_SD=float(prior_cv_ap.std()), CV_ROC_AUC=.5, Validation_AP=d['AP'], Validation_ROC_AUC=d['ROC_AUC'],
        Validation_Brier=d['Brier'], Validation_F1_threshold=f1_threshold(y.iloc[val], dummy.predict_proba(X.iloc[val])[:, 1]),
        Best_params='training class prior', Tuning_rows=len(tuning), Training_rows=len(train), Feature_count=0))
    validation = pd.DataFrame(rows).sort_values('Validation_AP', ascending=False)
    selected = validation.iloc[0].Model
    validation.to_csv(output / f'{name.lower()}_validation_comparison.csv', index=False)
    print(f'{name}: validation-selected model is {selected}; settings now frozen before test evaluation', flush=True)
    test_rows, predictions = [], pd.DataFrame({'row_index': X.index[test], 'group_id': groups.iloc[test].values, 'actual': y.iloc[test].values})
    for row in validation.itertuples():
        probability = fitted[row.Model].predict_proba(X.iloc[test])[:, 1]
        predictions[row.Model] = probability
        for label, threshold in [('0.5', .5), ('Validation-F1', row.Validation_F1_threshold)]:
            test_rows.append(dict(Dataset=name, Model=row.Model, Selected_before_test=row.Model == selected,
                Threshold_rule=label, **probability_scores(y.iloc[test], probability),
                **threshold_scores(y.iloc[test], probability, threshold), Test_rows=len(test),
                Test_prevalence=float(y.iloc[test].mean()), Feature_count=len(numeric)+len(categorical) if row.Model != 'Prior-probability baseline' else 0))
    test_table = pd.DataFrame(test_rows)
    test_table.to_csv(output / f'{name.lower()}_test_metrics.csv', index=False)
    predictions.to_csv(output / f'{name.lower()}_test_predictions.csv.gz', index=False, compression='gzip')
    importance = permutation_importance(fitted[selected], X.iloc[val], y.iloc[val], scoring='average_precision',
        n_repeats=5, random_state=SEED, n_jobs=1)
    pd.DataFrame({'Feature': numeric+categorical, 'Validation_AP_drop': importance.importances_mean,
        'Repeat_SD': importance.importances_std}).sort_values('Validation_AP_drop', ascending=False).to_csv(output / f'{name.lower()}_permutation_importance.csv', index=False)
    print(f'{name}: completed in {time.time()-started:.1f} seconds', flush=True)
    return test_table, validation


def run(root, output=None, max_cv_patients=5000, jobs=2):
    root = Path(root)
    stamp = datetime.now(ZoneInfo('Australia/Sydney')).strftime('%Y%m%d_%H%M%S')
    output = Path(output) if output else root / 'Model' / 'csiro_readmission_model_outputs' / stamp
    output.mkdir(parents=True, exist_ok=False)
    source = root / 'Model' / 'csiro_readmission_preparation_outputs' / 'csiro_inpatient_episodes.csv.gz'
    historical_path = root / 'EDA' / 'READMIT_3000_FE.xlsx'
    hashes = {str(f.relative_to(root)): hashlib.sha256(f.read_bytes()).hexdigest() for f in [source, historical_path]}
    c = pd.read_csv(source, dtype={'PATIENT': str})
    eligible = c.READMIT_30D_OBSERVED.notna()
    c = c.loc[eligible].copy().reset_index(drop=True)
    yc = c.READMIT_30D_OBSERVED.astype(int)
    Xc = c[C_NUM+C_CAT].copy()
    for col in C_NUM:
        Xc[col] = pd.to_numeric(Xc[col], errors='coerce')
    Xc['GENDER'] = Xc.GENDER.fillna('<missing>').astype(str)
    h = pd.read_excel(historical_path, engine='openpyxl')
    original = pd.read_excel(root / 'EDA' / 'READMIT_FE.xlsx', sheet_name='READMIT', engine='openpyxl')
    pd.testing.assert_frame_equal(h, original)
    if h.TEMPLATE_ID.isna().any() or h.TEMPLATE_ID.duplicated().any():
        raise ValueError('Historical profile IDs are not unique.')
    Xh, yh = h[H_NUM+H_CAT].copy(), pd.to_numeric(h.readmitted, errors='raise').astype(int)
    for col in H_NUM:
        Xh[col] = pd.to_numeric(Xh[col], errors='coerce')
    for col in H_CAT:
        Xh[col] = Xh[col].map(lambda v: '<missing>' if pd.isna(v) or str(v).strip() in ['', '.'] else str(v).strip())
    manifest = dict(seed=SEED, csiro_eligible_episodes=len(c), csiro_prevalence=float(yc.mean()),
        historical_profiles=len(h), historical_prevalence=float(yh.mean()),
        source_hashes=hashes, max_csiro_cv_patients=max_cv_patients, jobs=jobs,
        csiro_features=C_NUM+C_CAT, historical_features=H_NUM+H_CAT,
        csiro_target='30-day observed all-cause readmission; negative labels use unverified last-contact proxy',
        historical_target='source readmitted; horizon and planned/unplanned criteria unknown',
        pooling=False, prediction_time='discharge', csiro_los_at_least_365_days=int(c.CSIRO_EPISODE_LOS_DAYS.ge(365).sum()),
        warning='Exploratory LOS retained without clinical exclusion. Target, features and split units differ across datasets.')
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    ht, hv = fit_dataset('Historical', Xh, yh, h.TEMPLATE_ID.astype(str), H_NUM, H_CAT, output, jobs=jobs)
    ct, cv = fit_dataset('CSIRO', Xc, yc, c.PATIENT, C_NUM, C_CAT, output, max_cv_patients=max_cv_patients, jobs=jobs)
    combined = pd.concat([ht, ct], ignore_index=True)
    combined.to_csv(output / 'independent_dataset_comparison.csv', index=False)
    pd.concat([hv, cv], ignore_index=True).to_csv(output / 'validation_comparison.csv', index=False)
    assert all(hashlib.sha256((root / file).read_bytes()).hexdigest() == before for file, before in hashes.items())
    (output / 'COMPLETE.json').write_text(json.dumps({'completed': True, 'output': str(output)}), encoding='utf-8')
    print(combined.loc[combined.Selected_before_test & combined.Threshold_rule.eq('Validation-F1')].to_string(index=False), flush=True)
    print('OUTPUT:', output, flush=True)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--cv-patients', type=int, default=5000, help='0 means all training patients')
    parser.add_argument('--jobs', type=int, default=2)
    args = parser.parse_args()
    run(args.root, max_cv_patients=args.cv_patients or None, jobs=args.jobs)
