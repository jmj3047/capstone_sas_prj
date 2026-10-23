"""Leakage-safe admission-time LOS optimisation on the existing fixed split."""
from pathlib import Path
import hashlib
import json
import time

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder
from threadpoolctl import threadpool_limits

NUMERIC = ['PATIENT_AGE', 'NUM_CHRONIC_COND', 'NUM_VISITS']
CATEGORICAL = ['DOCTOR', 'DEPARTMENT', 'GENDER', 'Patient location',
               'RACE_CD', 'ICD9_TARGET', 'DIAGNOSIS_ICD_CODE']
FEATURES = NUMERIC + CATEGORICAL
TARGET = 'TOTAL_LOS'
SEED = 42


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_fixed_data(root, source_scope='all'):
    """Use precisely the preprocessing snapshot used by the previous comparison."""
    root = Path(root)
    previous = sorted((root / 'Model/external_los_comparison_outputs').glob('*/modelling_manifest.json'))
    if not previous:
        raise FileNotFoundError('Save the previous LOS comparison outputs first.')
    prior_dir = previous[-1].parent
    prior_manifest = json.loads(previous[-1].read_text(encoding='utf-8'))
    prep_dir = Path(prior_manifest['preprocessing_run'])
    if not prep_dir.exists():
        prep_dir = root / 'Model/external_los_preprocessing_outputs' / prep_dir.name
    prep_manifest = json.loads((prep_dir / 'preprocessing_manifest.json').read_text(encoding='utf-8'))
    if source_scope not in ('all', 'capstone'):
        raise ValueError('Unknown source validation scope: ' + str(source_scope))
    sources = prep_manifest['source_files']
    if source_scope == 'capstone':
        sources = {p: h for p, h in sources.items() if 'CAPSTONE' in Path(p).name}
        if len(sources) != 1:
            raise ValueError('Expected exactly one CAPSTONE source in the manifest.')
    for path, expected in sources.items():
        path = Path(path)
        if not path.exists() or sha256(path) != expected:
            raise ValueError('Input changed: ' + str(path) + '. Re-run preprocessing and baseline.')
    data_path = prep_dir / 'combined_los_model_table.csv'
    meta_path = prep_dir / 'combined_split_metadata.csv'
    if sha256(data_path) != prior_manifest['data_sha256'] or sha256(meta_path) != prior_manifest['metadata_sha256']:
        raise ValueError('The fixed data/split differs from the previous comparison.')
    data = pd.read_csv(data_path, dtype={c: str for c in CATEGORICAL}, keep_default_na=False, low_memory=False)
    for c in NUMERIC + [TARGET]:
        data[c] = pd.to_numeric(data[c], errors='coerce')
    for c in CATEGORICAL:
        data[c] = data[c].replace('', '<missing>')
    meta = pd.read_csv(meta_path, keep_default_na=False, low_memory=False)
    assert list(data) == FEATURES + [TARGET]
    assert len(data) == len(meta) and meta.ROW_ID.is_unique
    assert data[TARGET].notna().all() and np.isfinite(data[TARGET]).all()
    assert data[TARGET].ge(0).all()
    split_groups = {s: set(meta.loc[meta.SPLIT.eq(s), 'SPLIT_GROUP'])
                    for s in ['Train', 'Validation', 'Test']}
    for a, b in [('Train', 'Validation'), ('Train', 'Test'), ('Validation', 'Test')]:
        assert split_groups[a].isdisjoint(split_groups[b])
    prior_table = pd.read_csv(prior_dir / 'three_experiment_test_comparison.csv').set_index('Experiment')
    return data, meta, prior_table, prior_dir, prep_dir, prior_manifest


def add_features(frame, kind='original'):
    out = frame.copy()
    # A family is a nominal label, not an ordinal numerical code.
    out['ICD_FAMILY'] = out.DIAGNOSIS_ICD_CODE.str.split('.').str[0]
    if kind != 'original':
        age, chronic, visits = [pd.to_numeric(out[c], errors='coerce') for c in NUMERIC]
        out['LOG1P_VISITS'] = np.log1p(visits.clip(lower=0))
        out['AGE_CENTRED_SQUARED'] = ((age - 75) / 10) ** 2
        out['AGE_X_CHRONIC'] = (age / 10) * chronic
        out['CHRONIC_BAND'] = chronic.map(lambda v: '<missing>' if pd.isna(v) else str(int(min(v, 2))))
    return out


class HierarchicalMedianRegressor(RegressorMixin, BaseEstimator):
    """Training-only target medians with a minimum-size fallback."""
    def __init__(self, columns=('ICD_FAMILY',), min_count=30):
        self.columns = columns
        self.min_count = min_count

    def fit(self, X, y):
        frame = add_features(X)
        self.global_median_ = float(np.median(y))
        self.tables_ = []
        for columns in [self.columns[:n] for n in range(1, len(self.columns) + 1)]:
            grouped = frame[list(columns)].copy()
            grouped['_target'] = np.asarray(y)
            agg = grouped.groupby(list(columns), dropna=False)['_target'].agg(['size', 'median'])
            agg = agg.loc[agg['size'] >= self.min_count]
            table = {k if isinstance(k, tuple) else (k,): float(v)
                     for k, v in agg['median'].items()}
            self.tables_.append((columns, table))
        return self

    def predict(self, X):
        frame = add_features(X)
        prediction = np.full(len(frame), self.global_median_)
        for columns, table in self.tables_:
            keys = frame[list(columns)].itertuples(index=False, name=None)
            prediction = np.array([table.get(k, fallback) for k, fallback in zip(keys, prediction)])
        return prediction


SPECS = {
    'Train median': {'type': 'median'},
    'ICD-family median': {'type': 'group', 'columns': ('ICD_FAMILY',), 'min_count': 30},
    'Diagnosis-code median': {'type': 'group', 'columns': ('DIAGNOSIS_ICD_CODE',), 'min_count': 30},
    'ICD-family / department median': {'type': 'group', 'columns': ('ICD_FAMILY', 'DEPARTMENT'), 'min_count': 50},
    'HGB absolute / original / shallow': {'type': 'hgb', 'kind': 'original', 'loss': 'absolute_error', 'leaves': 7, 'min_leaf': 100},
    'HGB absolute / derived / shallow': {'type': 'hgb', 'kind': 'derived', 'loss': 'absolute_error', 'leaves': 7, 'min_leaf': 100},
    'HGB absolute / derived / regularised': {'type': 'hgb', 'kind': 'derived', 'loss': 'absolute_error', 'leaves': 15, 'min_leaf': 200},
    'HGB absolute / no doctor-location': {'type': 'hgb', 'kind': 'reduced', 'loss': 'absolute_error', 'leaves': 7, 'min_leaf': 100},
    'HGB log-target / derived': {'type': 'hgb', 'kind': 'derived', 'loss': 'squared_error', 'leaves': 7, 'min_leaf': 100, 'log_target': True},
}


def make_model(name):
    spec = SPECS[name]
    if spec['type'] == 'median':
        return DummyRegressor(strategy='median')
    if spec['type'] == 'group':
        return HierarchicalMedianRegressor(spec['columns'], spec['min_count'])
    numeric = NUMERIC.copy()
    categorical = CATEGORICAL.copy()
    if spec['kind'] != 'original':
        numeric += ['LOG1P_VISITS', 'AGE_CENTRED_SQUARED', 'AGE_X_CHRONIC']
        categorical += ['ICD_FAMILY', 'CHRONIC_BAND']
    if spec['kind'] == 'reduced':
        categorical = [c for c in categorical if c not in ['DOCTOR', 'Patient location']]
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)
    preprocessor = ColumnTransformer([
        ('numeric', SimpleImputer(strategy='median'), numeric),
        ('categorical', encoder, categorical),
    ], remainder='drop')
    # Ordinal codes are flagged as categorical to avoid learning arbitrary order.
    hgb = HistGradientBoostingRegressor(
        loss=spec['loss'], max_iter=180, learning_rate=.05,
        max_leaf_nodes=spec['leaves'], min_samples_leaf=spec['min_leaf'],
        l2_regularization=5, early_stopping=False,
        categorical_features=[False] * len(numeric) + [True] * len(categorical),
        random_state=SEED)
    pipe = Pipeline([
        ('features', FunctionTransformer(add_features, validate=False, kw_args={'kind': spec['kind']})),
        ('preprocess', preprocessor), ('model', hgb)
    ])
    if spec.get('log_target'):
        return TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1)
    return pipe


def metrics(y, pred):
    return {'MAE (days)': mean_absolute_error(y, pred),
            'RMSE (days)': np.sqrt(mean_squared_error(y, pred)), 'R2': r2_score(y, pred)}


def tune_on_validation(data, meta, run_cv=False, cv_folds=3):
    X, y = data[FEATURES], data[TARGET]
    train, val = meta.SPLIT.eq('Train'), meta.SPLIT.eq('Validation')
    masks = {'Historical baseline': train & meta.DATASET.eq('Historical'),
             'External-only': train & meta.DATASET.eq('External'), 'Combined': train}
    fitted, rows, chosen, cv_rows = {}, [], {}, []
    with threadpool_limits(limits=1):
        for experiment, mask in masks.items():
            for name in SPECS:
                t = time.perf_counter()
                print('FIT', experiment, '/', name, flush=True)
                model = make_model(name)
                model.fit(X.loc[mask], y.loc[mask])
                pred = model.predict(X.loc[val])
                assert np.isfinite(pred).all()
                fitted[(experiment, name)] = model
                rows.append({'Experiment': experiment, 'Candidate': name,
                             'Train rows': int(mask.sum()), **metrics(y.loc[val], pred),
                             'Fit seconds': time.perf_counter() - t})
                print('VAL MAE', round(rows[-1]['MAE (days)'], 4),
                      'seconds', round(rows[-1]['Fit seconds'], 1), flush=True)
            table = pd.DataFrame(rows)
            chosen[experiment] = table.loc[table.Experiment.eq(experiment)].sort_values(
                'MAE (days)', kind='stable').iloc[0].Candidate
            if run_cv:
                cv = cross_validate(make_model(chosen[experiment]), X.loc[mask], y.loc[mask],
                                    groups=meta.loc[mask, 'SPLIT_GROUP'],
                                    cv=GroupKFold(cv_folds), scoring='neg_mean_absolute_error',
                                    n_jobs=1, error_score='raise')
                cv_rows.append({'Experiment': experiment, 'Candidate': chosen[experiment],
                                'Train CV MAE': -cv['test_score'].mean(),
                                'Train CV SD': cv['test_score'].std(ddof=1)})
    return pd.DataFrame(rows), fitted, chosen, masks, pd.DataFrame(
        cv_rows, columns=['Experiment', 'Candidate', 'Train CV MAE', 'Train CV SD'])


def evaluate_selected(data, meta, prior, fitted, chosen):
    test = meta.SPLIT.eq('Test')
    test_y = data.loc[test, TARGET].to_numpy()
    test_meta = meta.loc[test].reset_index(drop=True)
    rows, domains, predictions, tails = [], [], {}, []
    with threadpool_limits(limits=1):
        for experiment, name in chosen.items():
            pred = fitted[(experiment, name)].predict(data.loc[test, FEATURES])
            assert np.isfinite(pred).all()
            predictions[experiment] = pred
            old_mae = float(prior.loc[experiment, 'Test MAE (days)'])
            score = metrics(test_y, pred)
            rows.append({'Experiment': experiment, 'Chosen candidate': name,
                         'Prior Test MAE': old_mae, **score,
                         'MAE reduction (days)': old_mae - score['MAE (days)'],
                         'MAE reduction (%)': 100 * (old_mae - score['MAE (days)']) / old_mae})
            for domain in ['Historical', 'External']:
                loc = test_meta.DATASET.eq(domain).to_numpy()
                domains.append({'Experiment': experiment, 'Domain': domain, 'Rows': int(loc.sum()),
                                **metrics(test_y[loc], pred[loc])})
            for label, loc in [('LOS < 15', test_y < 15), ('LOS >= 15', test_y >= 15)]:
                tails.append({'Experiment': experiment, 'Duration group': label, 'Rows': int(loc.sum()),
                              'MAE (days)': mean_absolute_error(test_y[loc], pred[loc]),
                              'Mean error (predicted - actual)': float((pred[loc] - test_y[loc]).mean())})
    return pd.DataFrame(rows).set_index('Experiment'), pd.DataFrame(domains), pd.DataFrame(tails), predictions
