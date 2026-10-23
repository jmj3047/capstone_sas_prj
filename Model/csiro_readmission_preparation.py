"""Read-only source audit and exploratory CSIRO inpatient readmission preparation.

No clinical crosswalk, unplanned-admission claim, or verified follow-up is implied.
"""
from pathlib import Path
from collections import Counter
import json
import numpy as np
import pandas as pd

SOURCE_URL = 'https://github.com/synthetichealth/synthea/wiki/CSV-File-Data-Dictionary'


def mapping_table():
    rows = [
        ('TEMPLATE_ID', 'PATIENT / episode ID', '식별자', '동일 인물이 아님. 조인/예측 변수로 사용 금지'),
        ('DOCTOR', 'PROVIDER', '비교 제외', '데이터별 식별자. 공통 의사 코드가 아님'),
        ('ADMIT_LOS', '(episode STOP - START) / 1 day', '파생·정의 차이', '관측 입원 episode 시간. ICU 포함 여부와 기존 정의 차이'),
        ('READMIT_LOS', 'next episode duration', '결과 정보', '재입원 후 정보. 예측 변수로 사용 금지'),
        ('readmitted', 'READMIT_7/30/90D_OBSERVED', '타겟 정렬 필요', '기존 관찰 기간 미확인. CSIRO는 계획/비계획 구분 불가'),
        ('ICU_DAYS', 'ICU encounter START / STOP', '조건부 파생 가능', '별도 ICU 입원 기록의 시간. 전체 ICU 시간 보장 안 됨. 이번에는 생성 안 함'),
        ('DEPARTMENT', 'providers.SPECIALITY', '직접 대응 불가', '현재 제공자 777명의 SPECIALITY가 모두 GENERAL PRACTICE'),
        ('DISCHARGED_TO', '', '없음', '퇴원 목적지 필드 없음'),
        ('NUM_CHRONIC_COND', 'conditions.START / STOP / CODE', '대체 지표', 'ACTIVE_CONDITION_CODES_AT_DISCHARGE 생성. 급성 질환 포함, 만성질환 수와 다름'),
        ('ORDER_SET_USED', '', '없음', '오더 세트 사용 여부 없음'),
        ('ORDER_TOTAL_CHARGES', 'TOTAL_CLAIM_COST', '정의 차이', 'encounter 전체 청구 비용. 기존 오더 비용 및 통화와 동일하다고 볼 수 없음'),
        ('GENDER', 'patients.GENDER', '직접 대응 후보', '원래 라벨 확인 후 공통 인코딩'),
        ('Patient location', 'CITY / COUNTY / STATE', '범주 정렬 필요', '호주/기존 지역 범주가 다름. 주소·우편번호 직접 조인 금지'),
        ('RACE_CD', 'patients.RACE', '범주 정렬 필요', '분류 체계와 코드북 대조 필요'),
        ('PATIENT_AGE', 'admission local date - BIRTHDATE', '파생 대응 후보', '입원 당시 나이. export 시점 나이 사용 금지'),
        ('DIAGNOSIS_ICD_GROUP', 'REASONCODE / conditions.CODE', '임상 매핑 필요', 'SNOMED에서 AMI/CHF/COPD/PNU로 검토된 매핑 필요'),
        ('ICD9_TARGET', '', '임상 매핑 필요', 'ICD-9 숫자 구간을 SNOMED 숫자에 적용 금지'),
        ('DRG_APR_CODE', '', '없음', 'APR-DRG 필드 없음'),
        ('DRG_APR_SEVERITY', '', '없음', 'APR 중증도 등급 없음'),
        ('DIAGNOSIS_ICD_CODE', 'REASONCODE / conditions.CODE', '코드 체계 다름', 'encounters.CODE는 방문 코드. 진단 코드로 사용 금지'),
        ('PROCEDURE_ICD_CODE', 'procedures.CODE', '코드 체계 다름', 'SNOMED procedure와 ICD-9 procedure 직접 동일시 불가'),
        ('DX_CODE', 'REASONCODE / conditions.CODE', '코드 체계 다름', '기존 진단 코드와 동일한 숫자 코드 아님'),
        ('OPERATION_COUNT', 'procedures rows', '직접 대응 불가', '약물 대조 등 비수술 기록 포함. 행 수를 수술 수로 바꾸지 않음'),
        ('NUM_VISITS', 'PRIOR_INPATIENT_EPISODES', '대체 지표', '이 파일에서 관측된 이전 입원 수. 기존 전체 방문 수와 다름'),
        ('TOTAL_LOS', 'CSIRO_EPISODE_LOS_DAYS', '정의 차이', '기존 ADMIT_LOS + ICU_DAYS와 다른 시간 기반 LOS. ICU를 다시 더하지 않음'),
    ]
    return pd.DataFrame(rows, columns=['Historical_column', 'CSIRO_source_or_feature', 'Status', 'Caution'])


def merge_episodes(inpatient, gap_hours=0):
    """Merge overlapping/touching intervals; optional transfer-gap sensitivity."""
    x = inpatient.sort_values(['PATIENT', 'START', 'STOP', 'Id']).copy()
    running_stop = x.groupby('PATIENT', sort=False)['STOP'].cummax()
    previous_stop = running_stop.groupby(x['PATIENT'], sort=False).shift()
    new = previous_stop.isna() | (x.START > previous_stop + pd.Timedelta(hours=gap_hours))
    x['EPISODE_NUMBER'] = new.groupby(x.PATIENT, sort=False).cumsum().astype(int)
    e = x.groupby(['PATIENT', 'EPISODE_NUMBER'], sort=False).agg(
        START=('START', 'min'), STOP=('STOP', 'max'),
        SOURCE_RECORD_COUNT=('Id', 'size'), FIRST_ENCOUNTER_ID=('Id', 'first'),
        FIRST_DESCRIPTION=('DESCRIPTION', 'first'),
        FIRST_REASONCODE=('REASONCODE', 'first'), FIRST_REASONDESCRIPTION=('REASONDESCRIPTION', 'first'),
    ).reset_index()
    e['PRIOR_INPATIENT_EPISODES'] = e.EPISODE_NUMBER - 1
    e['CSIRO_EPISODE_LOS_DAYS'] = (e.STOP - e.START).dt.total_seconds() / 86400
    e['NEXT_START'] = e.groupby('PATIENT', sort=False).START.shift(-1)
    e['GAP_TO_NEXT_DAYS'] = (e.NEXT_START - e.STOP).dt.total_seconds() / 86400
    return e


def local_day(s):
    return s.dt.tz_convert('Australia/Brisbane').dt.tz_localize(None).dt.normalize()


def long_stay_audit(inpatient, episodes, output):
    """365 days is an inspection threshold, not an automatic clinical exclusion."""
    output = Path(output)
    x = inpatient.copy()
    x['RAW_INPATIENT_LOS_DAYS'] = (x.STOP - x.START).dt.total_seconds() / 86400
    x.loc[x.RAW_INPATIENT_LOS_DAYS >= 365].sort_values('RAW_INPATIENT_LOS_DAYS', ascending=False).to_csv(output / 'long_raw_inpatient_records_audit.csv', index=False)
    episodes.loc[episodes.CSIRO_EPISODE_LOS_DAYS >= 365].sort_values('CSIRO_EPISODE_LOS_DAYS', ascending=False).to_csv(output / 'long_episodes_audit.csv', index=False)
    rows = []
    for name, duration in [('raw_inpatient_records', x.RAW_INPATIENT_LOS_DAYS), ('merged_episodes', episodes.CSIRO_EPISODE_LOS_DAYS)]:
        rows.append(dict(unit=name, count=len(duration), median_days=float(duration.median()),
            max_days=float(duration.max()), at_least_365_days=int(duration.ge(365).sum())))
    pd.DataFrame(rows).to_csv(output / 'los_quality_audit.csv', index=False)


def assign_target(e, horizon):
    """Observed all-cause readmission, with explicitly unverified contact-boundary proxy."""
    finish = e.STOP + pd.Timedelta(days=horizon)
    discharge_day, next_day = local_day(e.STOP), local_day(e.NEXT_START)
    end_day = local_day(finish)
    death = e.DEATHDATE
    death_at_index = death.notna() & (death <= discharge_day)
    uncertain_next = death.notna() & (death == next_day)
    next_before_death = death.isna() | (next_day < death)
    positive = (e.GAP_TO_NEXT_DAYS > 0) & (e.GAP_TO_NEXT_DAYS <= horizon) & next_before_death
    death_window = death.notna() & (death <= end_day)
    # A later recorded encounter is a proxy, NOT proof of continuous surveillance.
    negative = (e.LAST_RECORDED_CONTACT >= finish) & ~death_window & ~positive
    status = pd.Series('insufficient_followup_proxy', index=e.index)
    status.loc[negative] = 'negative_contact_proxy'
    status.loc[death_window & ~positive] = 'death_or_date_ambiguity'
    status.loc[positive] = 'positive_observed'
    status.loc[uncertain_next & (e.GAP_TO_NEXT_DAYS <= horizon)] = 'death_or_date_ambiguity'
    status.loc[death_at_index] = 'death_at_or_before_discharge'
    status.loc[e.AGE_AT_ADMISSION < 0] = 'invalid_negative_age'
    label = pd.Series(pd.NA, index=e.index, dtype='Int64')
    label.loc[status == 'positive_observed'] = 1
    label.loc[status == 'negative_contact_proxy'] = 0
    return label, status


def run(root, output, chunksize=250000, include_conditions=True):
    root, output = Path(root), Path(output)
    source = root / 'Data' / 'csiro_synthea_csv'
    output.mkdir(parents=True, exist_ok=True)
    schema = []
    for f in sorted(source.glob('*.csv')):
        columns = pd.read_csv(f, nrows=0).columns.tolist()
        schema.append(dict(file=f.name, size_bytes=f.stat().st_size,
                           mtime_ns=f.stat().st_mtime_ns, columns=', '.join(columns)))
    pd.DataFrame(schema).to_csv(output / 'source_inventory.csv', index=False)
    mapping_table().to_csv(output / 'feature_mapping.csv', index=False)
    p = pd.read_csv(source / 'patients.csv', dtype=str,
                    usecols=['Id', 'BIRTHDATE', 'DEATHDATE', 'GENDER', 'RACE', 'CITY', 'COUNTY', 'STATE'])
    if p.Id.duplicated().any():
        raise ValueError('Duplicate patient IDs: inspect before proceeding.')
    for col in ['BIRTHDATE', 'DEATHDATE']:
        p[col] = pd.to_datetime(p[col], errors='coerce')
    counts, selected, contact_parts = Counter(), [], []
    use = ['Id', 'PATIENT', 'START', 'STOP', 'ENCOUNTERCLASS', 'DESCRIPTION', 'REASONCODE', 'REASONDESCRIPTION']
    total = 0
    for chunk in pd.read_csv(source / 'encounters.csv', usecols=use, dtype=str, chunksize=chunksize):
        total += len(chunk)
        counts.update(chunk.ENCOUNTERCLASS.fillna('missing'))
        for col in ['START', 'STOP']:
            chunk[col] = pd.to_datetime(chunk[col], errors='coerce', utc=True)
        chunk['CONTACT_TIME'] = chunk[['START', 'STOP']].max(axis=1)
        contact_parts.append(chunk.groupby('PATIENT').CONTACT_TIME.max())
        selected.append(chunk.loc[chunk.ENCOUNTERCLASS.eq('inpatient'), use].copy())
        if total % 1000000 == 0:
            print(f'Scanned {total:,} encounters', flush=True)
    contact = pd.concat(contact_parts).groupby(level=0).max().rename('LAST_RECORDED_CONTACT')
    x = pd.concat(selected, ignore_index=True)
    bad = x.START.isna() | x.STOP.isna() | (x.STOP < x.START) | ~x.PATIENT.isin(p.Id)
    duplicates = x.Id.duplicated(keep=False)
    x.loc[bad | duplicates].to_csv(output / 'inpatient_excluded_audit.csv', index=False)
    # Ambiguous duplicate encounter IDs are excluded, not silently deduplicated.
    valid = x.loc[~bad & ~duplicates].copy()
    pd.Series(counts, name='records').rename_axis('encounter_class').to_csv(output / 'encounter_classes.csv')
    x.DESCRIPTION.value_counts(dropna=False).rename_axis('description').to_csv(output / 'inpatient_descriptions.csv')
    e = merge_episodes(valid).merge(p, left_on='PATIENT', right_on='Id', how='left', validate='many_to_one')
    e = e.merge(contact, left_on='PATIENT', right_index=True, validate='many_to_one')
    admission_day = local_day(e.START)
    e['AGE_AT_ADMISSION'] = (admission_day.dt.year - e.BIRTHDATE.dt.year -
        ((admission_day.dt.month < e.BIRTHDATE.dt.month) |
         ((admission_day.dt.month == e.BIRTHDATE.dt.month) & (admission_day.dt.day < e.BIRTHDATE.dt.day))).astype(int))
    long_stay_audit(valid, e, output)
    summaries = []
    for horizon in [7, 30, 90]:
        label, status = assign_target(e, horizon)
        e[f'READMIT_{horizon}D_OBSERVED'] = label
        e[f'READMIT_{horizon}D_STATUS'] = status
        for cohort, mask in [('all_ages', pd.Series(True, index=e.index)), ('age_18_plus', e.AGE_AT_ADMISSION >= 18), ('age_65_plus', e.AGE_AT_ADMISSION >= 65)]:
            y = label.loc[mask]
            summaries.append(dict(horizon_days=horizon, cohort=cohort, episodes=int(mask.sum()),
                eligible_proxy=int(y.notna().sum()), positive=int(y.eq(1).sum()), negative_proxy=int(y.eq(0).sum()),
                censored_or_excluded=int(y.isna().sum()), rate_among_eligible_proxy=float(y.mean()) if y.notna().any() else None))
    pd.DataFrame(summaries).to_csv(output / 'target_summary.csv', index=False)
    sensitivity = []
    for gap_hours in [0, 24]:
        s = merge_episodes(valid, gap_hours=gap_hours)
        sensitivity.append(dict(merge_gap_hours=gap_hours, episodes=len(s),
            observed_next_within_30d=int(s.GAP_TO_NEXT_DAYS.between(0, 30, inclusive='right').sum())))
    pd.DataFrame(sensitivity).to_csv(output / 'episode_merge_sensitivity.csv', index=False)
    if include_conditions:
        print('Scanning condition code dictionary; no ICD-9 crosswalk is invented.', flush=True)
        parts = []
        inpatient_patients = set(e.PATIENT)
        for c in pd.read_csv(source / 'conditions.csv', dtype=str, chunksize=chunksize):
            c = c.loc[c.PATIENT.isin(inpatient_patients), ['PATIENT', 'START', 'STOP', 'CODE', 'DESCRIPTION']].copy()
            parts.append(c)
        c = pd.concat(parts, ignore_index=True)
        c.groupby(['CODE', 'DESCRIPTION'], dropna=False).size().rename('condition_records').reset_index().to_csv(output / 'diagnosis_code_dictionary.csv', index=False)
        for col in ['START', 'STOP']:
            c[col] = pd.to_datetime(c[col], errors='coerce')
        condition_groups = {k: v for k, v in c.groupby('PATIENT', sort=False)}
        active = np.zeros(len(e), dtype=int)
        e['DISCHARGE_LOCAL_DATE'] = local_day(e.STOP)
        for patient, idx in e.groupby('PATIENT', sort=False).groups.items():
            cond = condition_groups.get(patient)
            if cond is None:
                continue
            for i in idx:
                day = e.at[i, 'DISCHARGE_LOCAL_DATE']
                mask = (cond.START <= day) & (cond.STOP.isna() | (cond.STOP > day))
                active[i] = cond.loc[mask, 'CODE'].nunique()
        e['ACTIVE_CONDITION_CODES_AT_DISCHARGE'] = active
    e.to_csv(output / 'csiro_inpatient_episodes.csv.gz', index=False, compression='gzip')
    e[['AGE_AT_ADMISSION', 'CSIRO_EPISODE_LOS_DAYS', 'PRIOR_INPATIENT_EPISODES']].describe().to_csv(output / 'derived_feature_summary.csv')
    report = dict(patients=len(p), encounters=total, inpatient_records=len(x), inpatient_patients=x.PATIENT.nunique(),
        invalid_inpatient_records=int(bad.sum()), ambiguous_duplicate_id_records=int(duplicates.sum()),
        merged_episodes=len(e), multi_record_episodes=int(e.SOURCE_RECORD_COUNT.gt(1).sum()),
        negative_age_episodes=int(e.AGE_AT_ADMISSION.lt(0).sum()),
        earliest_inpatient=str(x.START.min()), latest_inpatient=str(x.START.max()),
        last_observed_contact=str(contact.max()), confirmed_observation_end=None,
        source_url=SOURCE_URL, negative_label_rule='last recorded contact proxy, not verified follow-up',
        historical_readmission_window='unknown', unplanned_readmission_identifiable=False,
        condition_features_included=include_conditions)
    (output / 'audit_summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path)
    parser.add_argument('--skip-conditions', action='store_true')
    args = parser.parse_args()
    out = args.output or args.root / 'Model' / 'csiro_readmission_preparation_outputs'
    run(args.root, out, include_conditions=not args.skip_conditions)
