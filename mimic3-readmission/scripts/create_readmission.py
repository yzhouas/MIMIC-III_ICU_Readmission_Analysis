import argparse

from mimic3benchmark.util import *
import os
import sys


parser = argparse.ArgumentParser(description='Extract episodes from per-subject data.')
parser.add_argument('subjects_root_path', type=str, help='Directory containing subject sub-directories.')
parser.add_argument('--variable_map_file', type=str, default='resources/itemid_to_variable_map.csv',
                    help='CSV containing ITEMID-to-VARIABLE map.')
parser.add_argument('--reference_range_file', type=str, default='resources/variable_ranges.csv',
                    help='CSV containing reference ranges for VARIABLEs.')
parser.add_argument('--verbose', '-v', type=int, help='Level of verbosity in output.', default=1)
args, _ = parser.parse_known_args()


def merge_stays_counts(table1, table2):
    return table1.merge(table2, how='inner', left_on=['HADM_ID'], right_on=['HADM_ID'])

def add_inhospital_mortality_to_icustays(stays):
    mortality_all = stays.DOD.notnull() | stays.DEATHTIME.notnull()
    stays['MORTALITY'] = mortality_all.astype(int)

    mortality = stays.DEATHTIME.notnull() & ((stays.ADMITTIME <= stays.DEATHTIME) & (stays.DISCHTIME >= stays.DEATHTIME))
    #mortality = mortality | (stays.DEATHTIME.isnull() & stays.DOD.notnull() & ((stays.ADMITTIME <= stays.DOD) & (stays.DISCHTIME >= stays.DOD)))

    stays['MORTALITY0'] = mortality.astype(int)
    stays['MORTALITY_INHOSPITAL'] = stays['MORTALITY0']
    return stays


def add_inunit_mortality_to_icustays(stays):
    mortality = stays.DEATHTIME.notnull() & ((stays.INTIME <= stays.DEATHTIME) & (stays.OUTTIME >= stays.DEATHTIME))

    stays['MORTALITY_INUNIT'] = mortality.astype(int)
    return stays

def read_stays(subject_path):
    stays = dataframe_from_csv(os.path.join(subject_path, 'stays.csv'), index_col=None)
    stays.INTIME = pd.to_datetime(stays.INTIME)
    stays.OUTTIME = pd.to_datetime(stays.OUTTIME)
    stays.ADMITTIME = pd.to_datetime(stays.ADMITTIME)
    stays.DISCHTIME = pd.to_datetime(stays.DISCHTIME)
    stays.DOB = pd.to_datetime(stays.DOB)
    stays.DOD = pd.to_datetime(stays.DOD)
    stays.DEATHTIME = pd.to_datetime(stays.DEATHTIME)
    stays.sort_values(by=['INTIME', 'OUTTIME'], inplace=True)
    return stays
'''
stays = read_stays('/Users/jeffrey0925/MIMIC-III-clean/12607/')
stays=add_inhospital_mortality_to_icustays(stays)
stays=add_inunit_mortality_to_icustays(stays)
#stays=stays.drop(stays[(stays.MORTALITY==1)& (stays.MORTALITY_INHOSPITAL==1) & (stays.MORTALITY_INUNIT==1)].index)

counts=stays.groupby(['HADM_ID']).size().reset_index(name='COUNTS')
#print(counts)
stays=merge_stays_counts(stays,counts)
#print(stays)
max_outtimme=stays.groupby(['HADM_ID'])['OUTTIME'].transform(max)==stays['OUTTIME']
#print(max_outtimme)
stays['MAX_OUTTIME'] = max_outtimme.astype(int)
#print(stays)
transferback= (stays.COUNTS>1) & (stays.MAX_OUTTIME==0)
stays['TRANSFERBACK'] = transferback.astype(int)
#print(stays)
#----------------
dieinward=(stays.MORTALITY==1) & (stays.MORTALITY_INHOSPITAL==1) & (stays.MORTALITY_INUNIT==0)
stays['DIEINWARD'] = dieinward.astype(int)
#print(stays)
#----------------
next_admittime=stays[stays.groupby(['HADM_ID'])['OUTTIME'].transform(max)==stays['OUTTIME']]
next_admittime=next_admittime[['HADM_ID','ICUSTAY_ID','ADMITTIME','DISCHTIME']]
next_admittime['NEXT_ADMITTIME']=next_admittime.ADMITTIME.shift(-1)

next_admittime['DIFF']=next_admittime.NEXT_ADMITTIME-stays.DISCHTIME

stays=merge_stays_counts(stays,next_admittime[['HADM_ID','DIFF']])
less_than_30days=stays.DIFF.notnull() & (stays.DIFF<'30 days 00:00:00')
#print(less_than_30days)
stays['LESS_TAHN_30DAYS']=less_than_30days.astype(int)

#----------------

stays['DISCHARGE_DIE']=stays.DOD-stays.DISCHTIME

stays['DIE_LESS_TAHN_30DAYS']=(stays.MORTALITY==1) & (stays.MORTALITY_INHOSPITAL==0) & (stays.MORTALITY_INUNIT==0) & (stays.DISCHARGE_DIE<'30 days 00:00:00')
stays['DIE_LESS_TAHN_30DAYS']=stays['DIE_LESS_TAHN_30DAYS'].astype(int)
stays['READMISSION'] = ((stays.TRANSFERBACK==1) | (stays.DIEINWARD==1) | (stays.LESS_TAHN_30DAYS==1) | (stays.DIE_LESS_TAHN_30DAYS==1)).astype(int)
stays.ix[(stays.MORTALITY == 1) & (stays.MORTALITY_INHOSPITAL == 1) & (stays.MORTALITY_INUNIT == 1), 'READMISSION'] = 2
print(stays[['ICUSTAY_ID','MORTALITY','MORTALITY_INHOSPITAL','MORTALITY_INUNIT','TRANSFERBACK','DIEINWARD','LESS_TAHN_30DAYS','DIE_LESS_TAHN_30DAYS','READMISSION']])
#----------------
#stays['NEXT_ADMITTIME']=stays.ADMITTIME.shift(-1)
#print(stays)
#stays['DIFF']=stays.NEXT_ADMITTIME-stays.DISCHTIME
#print(stays)
#=========================
'''
for subject_dir in os.listdir(args.subjects_root_path):
    dn = os.path.join(args.subjects_root_path, subject_dir)
    try:
        subject_id = int(subject_dir)
        if not os.path.isdir(dn):
            raise Exception
    except:
        continue
    sys.stdout.write('Subject {}: '.format(subject_id))
    sys.stdout.flush()

    try:
        sys.stdout.write('reading...')
        sys.stdout.flush()
        stays = read_stays(os.path.join(args.subjects_root_path, subject_dir))
    except:
        sys.stdout.write('error reading from disk!\n')
        continue
    else:
        sys.stdout.write(
            'got {0} stays...'.format(stays.shape[0]))
        sys.stdout.flush()

    stays = add_inhospital_mortality_to_icustays(stays)
    stays = add_inunit_mortality_to_icustays(stays)
    #stays = stays.drop(stays[(stays.MORTALITY == 1) & (stays.MORTALITY_INHOSPITAL == 1) & (stays.MORTALITY_INUNIT == 1)].index)

    counts = stays.groupby(['HADM_ID']).size().reset_index(name='COUNTS')
    # print(counts)
    stays = merge_stays_counts(stays, counts)
    # print(stays)
    max_outtimme = stays.groupby(['HADM_ID'])['OUTTIME'].transform(max) == stays['OUTTIME']
    # print(max_outtimme)
    stays['MAX_OUTTIME'] = max_outtimme.astype(int)
    # print(stays)
    transferback = (stays.COUNTS > 1) & (stays.MAX_OUTTIME == 0)
    stays['TRANSFERBACK'] = transferback.astype(int)
    # print(stays)
    # ----------------
    dieinward = (stays.MORTALITY == 1) & (stays.MORTALITY_INHOSPITAL == 1) & (stays.MORTALITY_INUNIT == 0)
    stays['DIEINWARD'] = dieinward.astype(int)
    # print(stays)
    # ----------------
    next_admittime = stays[stays.groupby(['HADM_ID'])['OUTTIME'].transform(max) == stays['OUTTIME']]
    next_admittime = next_admittime[['HADM_ID', 'ICUSTAY_ID', 'ADMITTIME', 'DISCHTIME']]
    next_admittime['NEXT_ADMITTIME'] = next_admittime.ADMITTIME.shift(-1)

    next_admittime['DIFF'] = next_admittime.NEXT_ADMITTIME - stays.DISCHTIME

    stays = merge_stays_counts(stays, next_admittime[['HADM_ID', 'DIFF']])
    less_than_30days = stays.DIFF.notnull() & (stays.DIFF < '30 days 00:00:00')
    # print(less_than_30days)
    stays['LESS_TAHN_30DAYS'] = less_than_30days.astype(int)

    # ----------------

    stays['DISCHARGE_DIE'] = stays.DOD - stays.DISCHTIME

    stays['DIE_LESS_TAHN_30DAYS'] = (stays.MORTALITY == 1) & (stays.MORTALITY_INHOSPITAL == 0) & (stays.MORTALITY_INUNIT == 0) & (stays.DISCHARGE_DIE < '30 days 00:00:00')
    stays['DIE_LESS_TAHN_30DAYS'] = stays['DIE_LESS_TAHN_30DAYS'].astype(int)
    #print(stays[['ICUSTAY_ID', 'MORTALITY', 'MORTALITY_INHOSPITAL', 'MORTALITY_INUNIT', 'TRANSFERBACK', 'DIEINWARD','LESS_TAHN_30DAYS', 'DIE_LESS_TAHN_30DAYS']])
    # ----------------
    stays['READMISSION'] = ((stays.TRANSFERBACK==1) | (stays.DIEINWARD==1) | (stays.LESS_TAHN_30DAYS==1) | (stays.DIE_LESS_TAHN_30DAYS==1)).astype(int)

    stays.ix[(stays.MORTALITY == 1) & (stays.MORTALITY_INHOSPITAL == 1) & (stays.MORTALITY_INUNIT == 1), 'READMISSION'] = 2
    stays.to_csv(os.path.join(args.subjects_root_path, subject_dir, 'stays_readmission.csv'), index=False)

    sys.stdout.write(' DONE!\n')

#=========================
