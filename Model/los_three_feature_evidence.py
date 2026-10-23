"""Controlled predictive evidence for three timing-unconfirmed LOS features."""
import json
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits
import los_mae_optimisation as base
import los_advanced_fe_models as adv

EXTRAS = ['COMORBIDITY_INDEX', 'CARE_TEAM_SIZE', 'MONITORING_HOURS']


def load_external(root, source_scope='all'):
    data, metadata, prior, prior_dir, prep_dir, baseline_manifest = base.load_fixed_data(root, source_scope=source_scope)
    prep_manifest = json.loads((prep_dir / 'preprocessing_manifest.json').read_text(encoding='utf-8'))
    source = next(Path(p) for p in prep_manifest['source_files'] if 'CAPSTONE' in Path(p).name)
    raw = (pd.read_csv(source, low_memory=False) if source.suffix.lower() == '.csv' else
           pd.read_excel(source, sheet_name='table_CAPSTONE_2026_TRAIN_SET'))
    external = metadata.DATASET.eq('External')
    meta = metadata.loc[external].copy()
    indices = pd.to_numeric(meta.SOURCE_ROW).astype(int) - 2
    aligned = raw.iloc[indices].copy()
    aligned.index = meta.index
    assert (aligned.ENCOUNTER_KEY.astype(str).values ==
            meta.ENCOUNTER_KEY.astype(str).str.replace(r'\.0$', '', regex=True).values).all()
    X = data.loc[external, base.FEATURES].copy()
    for col in EXTRAS:
        X[col] = pd.to_numeric(aligned[col], errors='coerce')
    y = data.loc[external, base.TARGET].copy()
    assert np.allclose(y, pd.to_numeric(aligned.ADMIT_LOS) + pd.to_numeric(aligned.ICU_DAYS))
    group_sets = [set(meta.loc[meta.SPLIT.eq(s), 'SPLIT_GROUP'])
                  for s in ['Train', 'Validation', 'Test']]
    assert all(group_sets[i].isdisjoint(group_sets[j]) for i in range(3) for j in range(i+1, 3))
    return data, metadata, X, y, meta, prior_dir, prep_dir, baseline_manifest


def paired_interval(error_difference, groups, draws=1000, seed=42):
    """Cluster-bootstrap paired mean-error difference, not model-selection uncertainty."""
    frame = pd.DataFrame({'Group': np.asarray(groups), 'Difference': error_difference})
    agg = frame.groupby('Group').Difference.agg(['sum', 'size'])
    sums, sizes = agg['sum'].to_numpy(), agg['size'].to_numpy()
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(draws):
        indices = rng.integers(0, len(sums), len(sums))
        values.append(sums[indices].sum() / sizes[indices].sum())
    low, high = np.quantile(values, [.025, .975])
    return float(np.mean(error_difference)), float(low), float(high)


def run_evidence(root, repeats=20, bootstrap_draws=1000):
    data, metadata, X, y, meta, prior_dir, prep_dir, baseline_manifest = load_external(root)
    subsets = {'Common only': []}
    for size in range(1, 4):
        for combination in combinations(EXTRAS, size):
            label = 'All three' if size == 3 else ' + '.join(combination)
            subsets[label] = list(combination)
    print('Fitting all 8 feature subsets on the same training rows.', flush=True)
    table, correlations, models = adv.conditional_comorbidity_experiment(
        root, data, metadata, prep_dir, feature_sets=subsets)
    train, val, test = [meta.SPLIT.eq(s) for s in ['Train', 'Validation', 'Test']]
    test_y = y.loc[test].to_numpy()
    predictions = {}
    with threadpool_limits(limits=1):
        for label in subsets:
            predictions[label] = models[label].predict(X.loc[test])
    control_error = np.abs(test_y - predictions['Common only'])
    full_error = np.abs(test_y - predictions['All three'])
    test_groups = meta.loc[test, 'SPLIT_GROUP'].to_numpy()
    gains = []
    for label in subsets:
        difference = control_error - np.abs(test_y - predictions[label])
        estimate, low, high = paired_interval(difference, test_groups, bootstrap_draws)
        gains.append({'Experiment': label, 'MAE reduction vs common (days)': estimate,
                      'Approx 95% lower': low, 'Approx 95% upper': high})
    table = table.merge(pd.DataFrame(gains), on='Experiment', validate='one_to_one')
    table['MAE reduction vs common (%)'] = 100 * table['MAE reduction vs common (days)'] / control_error.mean()
    loo = []
    for feature in EXTRAS:
        subset = [f for f in EXTRAS if f != feature]
        label = next(k for k, v in subsets.items() if v == subset)
        difference = np.abs(test_y - predictions[label]) - full_error
        estimate, low, high = paired_interval(difference, test_groups, bootstrap_draws)
        loo.append({'Removed feature': feature, 'MAE without feature': float(np.abs(test_y-predictions[label]).mean()),
                    'Full model MAE': float(full_error.mean()), 'MAE increase after refitting (days)': estimate,
                    'Approx 95% lower': low, 'Approx 95% upper': high})
    loo = pd.DataFrame(loo)

    # Held-out validation importance: refitting is not performed here.
    print('Validation permutation importance (individual and joint).', flush=True)
    full = models['All three']
    X_val, y_val = X.loc[val].copy(), y.loc[val].to_numpy()
    val_error = base.metrics(y_val, full.predict(X_val))['MAE (days)']
    perm_rows = []
    rng = np.random.default_rng(base.SEED)
    with threadpool_limits(limits=1):
        for label, cols in [(c, [c]) for c in EXTRAS] + [('All three jointly', EXTRAS)]:
            values = []
            for _ in range(repeats):
                shuffled = X_val.copy()
                permutation = rng.permutation(len(X_val))
                # One common permutation preserves correlations within the block.
                shuffled[cols] = X_val[cols].to_numpy()[permutation]
                values.append(base.metrics(y_val, full.predict(shuffled))['MAE (days)'] - val_error)
            perm_rows.append({'Feature/block': label, 'Validation MAE increase (days)': np.mean(values),
                              'Across-repeat SD': np.std(values, ddof=1), 'Repeats': repeats})
    permutation_table = pd.DataFrame(perm_rows)

    # Predefined main comparison only; folds remain entirely within original Train.
    print('Train-only 3-fold group CV: common vs all-three.', flush=True)
    cv = GroupKFold(n_splits=3)
    cv_rows = []
    X_train, y_train, g_train = X.loc[train], y.loc[train], meta.loc[train, 'SPLIT_GROUP']
    with threadpool_limits(limits=1):
        for fold, (fit_index, hold_index) in enumerate(cv.split(X_train, y_train, g_train), 1):
            for label in ['Common only', 'All three']:
                estimator = clone(models[label])
                estimator.fit(X_train.iloc[fit_index], y_train.iloc[fit_index])
                pred = estimator.predict(X_train.iloc[hold_index])
                cv_rows.append({'Fold': fold, 'Experiment': label,
                                **base.metrics(y_train.iloc[hold_index], pred)})
            print('CV fold', fold, 'complete.', flush=True)
    cv_table = pd.DataFrame(cv_rows)
    train_correlations = X.loc[train, EXTRAS].corr(method='spearman')
    test_meta = meta.loc[test, ['ROW_ID', 'DATASET', 'SPLIT_GROUP']].reset_index(drop=True)
    test_meta['ACTUAL_TOTAL_LOS'] = test_y
    for label, pred in predictions.items():
        test_meta[label + ' prediction'] = pred
    availability = pd.DataFrame({
        'Feature': EXTRAS, 'Definition confirmed': ['No']*3,
        'Available at admission confirmed': ['No']*3,
        'Generated without LOS confirmed': ['No']*3,
        'Status': ['Candidate only; requires provider confirmation']*3
    })
    return {
        'ablation': table, 'leave_one_out': loo, 'permutation': permutation_table,
        'train_cv': cv_table, 'train_feature_correlations': train_correlations,
        'train_target_correlations': correlations.reset_index(names='Feature'), 'availability_gate': availability,
        'test_predictions': test_meta,
        'manifest': {'baseline_run': str(prior_dir), 'preprocessing_run': str(prep_dir),
                     'seed': base.SEED, 'subsets': subsets, 'permutation_repeats': repeats,
                     'bootstrap_draws': bootstrap_draws, 'train_cv_folds': 3,
                     'test_status': 'Previously inspected test; exploratory evidence, not independent confirmation',
                     'primary_status': 'Extended candidate; admission timing and target-independent generation unconfirmed',
                     'source_files': baseline_manifest['source_files'] if 'source_files' in baseline_manifest else
                         json.loads((prep_dir/'preprocessing_manifest.json').read_text())['source_files'],
                     'helper_sha256': base.sha256(Path(root)/'Model/los_three_feature_evidence.py')}
    }
