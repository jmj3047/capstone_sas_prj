"""Additional admission-time LOS model/feature experiments on a frozen split."""
import time
import warnings
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, HuberRegressor, QuantileRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
import los_mae_optimisation as base


def advanced_features(frame, interactions=True):
    out = base.add_features(frame, kind='derived')
    age, chronic, visits = [pd.to_numeric(out[c], errors='coerce')
                           for c in base.NUMERIC]
    out['CHRONIC_X_VISITS'] = chronic * visits
    out['AGE_X_LOG_VISITS'] = (age / 10) * np.log1p(visits.clip(lower=0))
    out['AGE_BAND'] = age.map(lambda v: '<missing>' if pd.isna(v) else str(int(v // 10)))
    out['VISIT_BAND'] = visits.map(lambda v: '<missing>' if pd.isna(v) else
                                 '0' if v == 0 else '1' if v == 1 else '2-3' if v <= 3 else '4+')
    if interactions:
        for name, a, b in [
            ('ICD_X_DEPT', 'ICD_FAMILY', 'DEPARTMENT'),
            ('ICD_X_CHRONIC', 'ICD_FAMILY', 'CHRONIC_BAND'),
            ('ICD_X_AGE_BAND', 'ICD_FAMILY', 'AGE_BAND'),
            ('DEPT_X_GENDER', 'DEPARTMENT', 'GENDER')
        ]:
            out[name] = out[a].astype(str) + '::' + out[b].astype(str)
    return out


EXTRA_NUM = ['LOG1P_VISITS', 'AGE_CENTRED_SQUARED', 'AGE_X_CHRONIC',
             'CHRONIC_X_VISITS', 'AGE_X_LOG_VISITS']
EXTRA_CAT = ['ICD_FAMILY', 'CHRONIC_BAND', 'AGE_BAND', 'VISIT_BAND']
CROSSES = ['ICD_X_DEPT', 'ICD_X_CHRONIC', 'ICD_X_AGE_BAND', 'DEPT_X_GENDER']

SPECS = {
    'HGB FE / 3 leaves / smooth': {'type': 'hgb', 'leaves': 3, 'min_leaf': 500, 'iterations': 350, 'lr': .05},
    'HGB FE / 7 leaves / slow': {'type': 'hgb', 'leaves': 7, 'min_leaf': 100, 'iterations': 500, 'lr': .03},
    'HGB FE / 7 leaves / smooth': {'type': 'hgb', 'leaves': 7, 'min_leaf': 500, 'iterations': 300, 'lr': .05},
    'HGB FE / 15 leaves': {'type': 'hgb', 'leaves': 15, 'min_leaf': 200, 'iterations': 350, 'lr': .03},
    'HGB FE / 31 leaves': {'type': 'hgb', 'leaves': 31, 'min_leaf': 100, 'iterations': 300, 'lr': .03},
    'HGB FE / pairwise': {'type': 'hgb', 'leaves': 7, 'min_leaf': 100, 'iterations': 400, 'lr': .03, 'interaction': 'pairwise'},
    'HGB FE / main effects only': {'type': 'hgb', 'leaves': 7, 'min_leaf': 200, 'iterations': 400, 'lr': .03, 'interaction': 'no_interactions', 'crosses': False},
    'HGB FE / log target': {'type': 'hgb', 'leaves': 7, 'min_leaf': 200, 'iterations': 350, 'lr': .03, 'log_target': True},
    'Extra Trees / log target / leaf 30': {'type': 'extra', 'min_leaf': 30, 'log_target': True},
    'Extra Trees / log target / leaf 100': {'type': 'extra', 'min_leaf': 100, 'log_target': True},
    'Random Forest / log target': {'type': 'forest', 'min_leaf': 50, 'log_target': True},
    'Ridge FE / log target': {'type': 'ridge', 'log_target': True},
    'Huber FE / log target': {'type': 'huber', 'log_target': True},
    'Quantile linear / alpha .01 / train cap': {'type': 'quantile', 'alpha': .01, 'max_rows': 20000, 'crosses': False},
    'Quantile linear / alpha .05 / train cap': {'type': 'quantile', 'alpha': .05, 'max_rows': 20000, 'crosses': False},
}


def make_model(name):
    spec = SPECS[name]
    numeric = base.NUMERIC + EXTRA_NUM
    categorical = [c for c in base.CATEGORICAL if c not in ['DOCTOR', 'Patient location']]
    categorical += EXTRA_CAT + (CROSSES if spec.get('crosses', True) else [])
    kind = spec['type']
    num_steps = [('impute', SimpleImputer(strategy='median'))]
    if kind in ['ridge', 'huber', 'quantile']:
        num_steps.append(('scale', StandardScaler()))
    if kind == 'hgb':
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan,
                                 min_frequency=20, max_categories=200)
        estimator = HistGradientBoostingRegressor(
            loss='squared_error' if spec.get('log_target') else 'absolute_error',
            max_iter=spec['iterations'], learning_rate=spec['lr'],
            max_leaf_nodes=spec['leaves'], min_samples_leaf=spec['min_leaf'],
            l2_regularization=10, early_stopping=False,
            categorical_features=[False] * len(numeric) + [True] * len(categorical),
            interaction_cst=spec.get('interaction'), random_state=base.SEED)
    else:
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False,
                                min_frequency=30, max_categories=200)
        if kind == 'extra':
            estimator = ExtraTreesRegressor(n_estimators=120, min_samples_leaf=spec['min_leaf'],
                                           max_depth=16, max_features=.7, n_jobs=2, random_state=base.SEED)
        elif kind == 'forest':
            estimator = RandomForestRegressor(n_estimators=100, min_samples_leaf=spec['min_leaf'],
                                             max_depth=14, max_features=.7, max_samples=.65,
                                             n_jobs=2, random_state=base.SEED)
        elif kind == 'ridge':
            estimator = Ridge(alpha=100)
        elif kind == 'huber':
            estimator = HuberRegressor(epsilon=1.35, alpha=.1, max_iter=400)
        else:
            estimator = QuantileRegressor(quantile=.5, alpha=spec['alpha'], solver='highs',
                                          solver_options={'time_limit': 60})
    prep = ColumnTransformer([
        ('numeric', Pipeline(num_steps), numeric), ('categorical', encoder, categorical)
    ], remainder='drop')
    pipe = Pipeline([
        ('features', FunctionTransformer(advanced_features, validate=False,
                                        kw_args={'interactions': spec.get('crosses', True)})),
        ('preprocess', prep), ('model', estimator)
    ])
    if spec.get('log_target'):
        return TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1)
    return pipe


def load_reference(root):
    results = sorted((Path(root) / 'Model/los_mae_optimisation_outputs').glob('*/optimisation_manifest.json'))
    if not results:
        raise FileNotFoundError('Run the previous MAE optimisation and save results first.')
    path = results[-1]
    manifest = json.loads(path.read_text(encoding='utf-8'))
    return path.parent, manifest


def capped_train_indices(indices, meta, max_rows):
    if max_rows is None or len(indices) <= max_rows:
        return np.asarray(indices)
    # Whole groups only. Never sample validation or test rows for fitting.
    rng = np.random.default_rng(base.SEED)
    group_rows = meta.loc[indices].groupby('SPLIT_GROUP', sort=False).groups
    group_ids = list(group_rows)
    rng.shuffle(group_ids)
    out = []
    for group_id in group_ids:
        rows = list(group_rows[group_id])
        if len(out) + len(rows) <= max_rows:
            out.extend(rows)
    return np.asarray(out)


def search(data, meta, reference_manifest, experiments=None):
    experiments = experiments or ['Historical baseline', 'External-only', 'Combined']
    X, y = data[base.FEATURES], data[base.TARGET]
    val = meta.SPLIT.eq('Validation')
    masks = {'Historical baseline': meta.SPLIT.eq('Train') & meta.DATASET.eq('Historical'),
             'External-only': meta.SPLIT.eq('Train') & meta.DATASET.eq('External'),
             'Combined': meta.SPLIT.eq('Train')}
    fitted, predictions, rows, choices = {}, {}, [], {}
    with threadpool_limits(limits=1):
        for experiment in experiments:
            indices = meta.index[masks[experiment]]
            names = ['Previous selected model'] + list(SPECS)
            for name in names:
                t = time.perf_counter()
                print('FIT', experiment, '/', name, flush=True)
                actual_indices = (np.asarray(indices) if name == 'Previous selected model' else
                                  capped_train_indices(indices, meta, SPECS[name].get('max_rows')))
                try:
                    model = (base.make_model(reference_manifest['chosen'][experiment])
                             if name == 'Previous selected model' else make_model(name))
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter('always', ConvergenceWarning)
                        model.fit(X.loc[actual_indices], y.loc[actual_indices])
                    convergence = [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]
                    if convergence:
                        raise RuntimeError('Model did not converge: ' + '; '.join(convergence))
                    prediction = model.predict(X.loc[val])
                    if not np.isfinite(prediction).all():
                        raise ValueError('Non-finite validation predictions.')
                    fitted[(experiment, name)] = model
                    predictions[(experiment, name)] = prediction
                    rows.append({'Experiment': experiment, 'Candidate': name,
                                 'Actual train rows': len(actual_indices), 'Status': 'OK',
                                 **base.metrics(y.loc[val], prediction),
                                 'Fit seconds': time.perf_counter() - t})
                    print('VAL MAE', round(rows[-1]['MAE (days)'], 4),
                          'seconds', round(rows[-1]['Fit seconds'], 1), flush=True)
                except (ValueError, RuntimeError, TypeError) as error:
                    rows.append({'Experiment': experiment, 'Candidate': name,
                                 'Actual train rows': len(actual_indices), 'Status': str(error),
                                 'MAE (days)': np.nan, 'RMSE (days)': np.nan, 'R2': np.nan,
                                 'Fit seconds': time.perf_counter() - t})
                    print('FAILED', str(error)[:180], flush=True)
            ok = pd.DataFrame(rows)
            ok = ok[ok.Experiment.eq(experiment) & ok.Status.eq('OK')]
            if ok.empty:
                raise RuntimeError('No model succeeded for ' + experiment)
            best = ok.sort_values('MAE (days)', kind='stable').iloc[0].Candidate
            previous_val = predictions[(experiment, 'Previous selected model')]
            best_val = predictions[(experiment, best)]
            weights = [0, .25, .5, .75, 1]
            ensemble_maes = [base.metrics(y.loc[val], (1-w) * previous_val + w * best_val)['MAE (days)']
                            for w in weights]
            weight = weights[int(np.argmin(ensemble_maes))]
            choices[experiment] = {'candidate': best, 'new_weight': weight,
                                   'validation_mae': min(ensemble_maes)}
            print('LOCKED', experiment, choices[experiment], flush=True)
    return pd.DataFrame(rows), fitted, choices, masks


def predict_choice(fitted, choices, experiment, X):
    choice = choices[experiment]
    previous = fitted[(experiment, 'Previous selected model')].predict(X)
    best = fitted[(experiment, choice['candidate'])].predict(X)
    return (1-choice['new_weight']) * previous + choice['new_weight'] * best


def conditional_comorbidity_experiment(root, data, meta, prep_dir, feature_sets=None):
    """Exploratory only: admission-time/provenance status of extra fields is unknown.

    This does not modify or replace the primary common-feature experiments.
    """
    prep_manifest = json.loads((Path(prep_dir) / 'preprocessing_manifest.json').read_text(encoding='utf-8'))
    source_path = next(Path(p) for p in prep_manifest['source_files'] if 'CAPSTONE' in Path(p).name)
    raw = (pd.read_csv(source_path, low_memory=False) if source_path.suffix.lower() == '.csv' else
           pd.read_excel(source_path, sheet_name='table_CAPSTONE_2026_TRAIN_SET'))
    external = meta.DATASET.eq('External')
    source_rows = pd.to_numeric(meta.loc[external, 'SOURCE_ROW']).astype(int) - 2
    aligned = raw.iloc[source_rows].copy()
    aligned.index = meta.index[external]
    expected_y = pd.to_numeric(aligned.ADMIT_LOS) + pd.to_numeric(aligned.ICU_DAYS)
    assert np.allclose(expected_y, data.loc[external, base.TARGET])
    assert (aligned.ENCOUNTER_KEY.astype(str).values ==
            meta.loc[external, 'ENCOUNTER_KEY'].astype(str).str.replace(r'\.0$', '', regex=True).values).all()
    X = data.loc[external, base.FEATURES].copy()
    audit_cols = ['MONITORING_HOURS', 'COMORBIDITY_INDEX', 'CARE_TEAM_SIZE']
    for col in audit_cols:
        X[col] = pd.to_numeric(aligned[col], errors='coerce')
    y = data.loc[external, base.TARGET]
    train = meta.loc[external, 'SPLIT'].eq('Train')
    val = meta.loc[external, 'SPLIT'].eq('Validation')
    test = meta.loc[external, 'SPLIT'].eq('Test')
    audit = aligned.loc[train, audit_cols].apply(pd.to_numeric, errors='coerce')
    audit[base.TARGET] = y.loc[train]
    correlations = audit.corr(method='spearman')[base.TARGET].drop(base.TARGET).to_frame('Train-only Spearman rho')
    rows, fitted = [], {}
    feature_sets = feature_sets or {
        'Common features only': [],
        'Common + COMORBIDITY_INDEX (unconfirmed timing)': ['COMORBIDITY_INDEX']
    }
    if any(col not in audit_cols for cols in feature_sets.values() for col in cols):
        raise ValueError('Unknown extra feature requested.')
    with threadpool_limits(limits=1):
        for name, extras in feature_sets.items():
            numeric = base.NUMERIC + ['LOG1P_VISITS', 'AGE_CENTRED_SQUARED', 'AGE_X_CHRONIC']
            numeric += extras
            categorical = [c for c in base.CATEGORICAL if c not in ['DOCTOR', 'Patient location']]
            categorical += ['ICD_FAMILY', 'CHRONIC_BAND']
            preprocessor = ColumnTransformer([
                ('numeric', SimpleImputer(strategy='median'), numeric),
                ('categorical', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan), categorical)
            ], remainder='drop')
            estimator = HistGradientBoostingRegressor(
                loss='absolute_error', max_iter=180, learning_rate=.05, max_leaf_nodes=7,
                min_samples_leaf=100, l2_regularization=5, early_stopping=False,
                categorical_features=[False]*len(numeric) + [True]*len(categorical), random_state=base.SEED)
            pipe = Pipeline([
                ('features', FunctionTransformer(base.add_features, validate=False, kw_args={'kind': 'reduced'})),
                ('preprocess', preprocessor), ('model', estimator)
            ])
            pipe.fit(X.loc[train], y.loc[train])
            pred_val, pred_test = pipe.predict(X.loc[val]), pipe.predict(X.loc[test])
            rows.append({'Experiment': name, 'Added features': ', '.join(extras) or 'None',
                         'Train rows': int(train.sum()), 'Validation rows': int(val.sum()),
                         'Test rows': int(test.sum()),
                         'Validation MAE': base.metrics(y.loc[val], pred_val)['MAE (days)'],
                         **base.metrics(y.loc[test], pred_test),
                         'Use status': 'Exploratory only; timing and target-independent generation unconfirmed'})
            long_stay = y.loc[test].ge(15).to_numpy()
            rows[-1]['Long-stay rows (LOS >= 15)'] = int(long_stay.sum())
            rows[-1]['Long-stay MAE'] = base.metrics(
                y.loc[test].to_numpy()[long_stay], pred_test[long_stay])['MAE (days)']
            fitted[name] = pipe
    return pd.DataFrame(rows), correlations, fitted
