from __future__ import print_function

import argparse
import yaml

from mimic3benchmark.mimic3csv import *
from mimic3benchmark.preprocessing import add_hcup_ccs_2015_groups, make_phenotype_label_matrix
from mimic3benchmark.util import *

parser = argparse.ArgumentParser(description='Extract per-subject data from MIMIC-III CSV files.')
parser.add_argument('mimic3_path', type=str, help='Directory containing MIMIC-III CSV files.')
parser.add_argument('output_path', type=str, help='Directory where per-subject data should be written.')
parser.add_argument('--event_tables', '-e', type=str, nargs='+', help='Tables from which to read events.',default=['CHARTEVENTS', 'LABEVENTS', 'OUTPUTEVENTS'])
parser.add_argument('--phenotype_definitions', '-p', type=str, default='resources/hcup_ccs_2015_definitions.yaml',
                    help='YAML file with phenotype definitions.')
parser.add_argument('--itemids_file', '-i', type=str, help='CSV containing list of ITEMIDs to keep.')
parser.add_argument('--verbose', '-v', type=int, help='Level of verbosity in output.', default=1)
parser.add_argument('--test', action='store_true', help='TEST MODE: process only 1000 subjects, 1000000 events.')
args, _ = parser.parse_known_args()

try:
    os.makedirs(args.output_path)
except:
    pass

patients = read_patients_table(args.mimic3_path)
admits = read_admissions_table(args.mimic3_path)

transfers = read_transfers_table(args.mimic3_path)
if args.verbose:
    print('transfers_START:', transfers.ICUSTAY_ID.unique().shape[0], transfers.HADM_ID.unique().shape[0],
          transfers.SUBJECT_ID.unique().shape[0])

stays = read_icustays_table(args.mimic3_path)

if args.verbose:
    print('stays_START:', stays.ICUSTAY_ID.unique().shape[0], stays.HADM_ID.unique().shape[0],
          stays.SUBJECT_ID.unique().shape[0])

transfers = merge_on_subject_admission(transfers, admits)
transfers = merge_on_subject(transfers, patients)

stays = merge_on_subject_admission(stays, admits)
stays = merge_on_subject(stays, patients)

transfers = add_age_to_icustays(transfers)
transfers = add_inunit_mortality_to_icustays(transfers)
transfers = add_inhospital_mortality_to_icustays(transfers)
transfers = filter_icustays_on_age(transfers)
if args.verbose:
    print('transfers_REMOVE PATIENTS AGE < 18:', transfers.ICUSTAY_ID.unique().shape[0], transfers.HADM_ID.unique().shape[0],
          transfers.SUBJECT_ID.unique().shape[0])

stays = add_age_to_icustays(stays)
stays = add_inunit_mortality_to_icustays(stays)
stays = add_inhospital_mortality_to_icustays(stays)
stays = filter_icustays_on_age(stays)
if args.verbose:
    print('stays_REMOVE PATIENTS AGE < 18:', stays.ICUSTAY_ID.unique().shape[0], stays.HADM_ID.unique().shape[0],
          stays.SUBJECT_ID.unique().shape[0])

transfers.to_csv(os.path.join(args.output_path, 'all_transfers.csv'), index=False)
print ('stransfers_done')
stays.to_csv(os.path.join(args.output_path, 'all_stays.csv'), index=False)
print ('stays_done')
#====================================================================================

diagnoses = read_icd_diagnoses_table(args.mimic3_path)
diagnoses = filter_diagnoses_on_stays(diagnoses, stays)
diagnoses.to_csv(os.path.join(args.output_path, 'all_diagnoses.csv'), index=False)
print ('all_diagnoses_done')
count_icd_codes(diagnoses, output_path=os.path.join(args.output_path, 'diagnosis_counts.csv'))
print ('diagnosis_counts_done')
#====================================================================================

procedures = read_icd_procedures_table(args.mimic3_path)
procedures = filter_diagnoses_on_stays(procedures, stays)
procedures.to_csv(os.path.join(args.output_path, 'all_procedures.csv'), index=False)
print ('all_procedures_done')
count_icd_codes(procedures, output_path=os.path.join(args.output_path, 'procedures_counts.csv'))
print ('procedures_counts_done')
#----------
prescriptions = read_prescriptions_table(args.mimic3_path)
prescriptions.to_csv(os.path.join(args.output_path, 'all_prescriptions.csv'), index=False)
print ('all_prescriptions_done')

#====================================================================================
phenotypes = add_hcup_ccs_2015_groups(diagnoses, yaml.load(open(args.phenotype_definitions, 'r')))
make_phenotype_label_matrix(phenotypes, stays).to_csv(os.path.join(args.output_path, 'phenotype_labels.csv'),
                                                      index=False, quoting=csv.QUOTE_NONNUMERIC)
#====================================================================================

subjects = stays.SUBJECT_ID.unique()
break_up_stays_by_subject(stays, args.output_path, subjects=subjects, verbose=args.verbose)
break_up_transfers_by_subject(transfers, args.output_path, subjects=subjects, verbose=args.verbose)

break_up_diagnoses_by_subject(phenotypes, args.output_path, subjects=subjects, verbose=args.verbose)
break_up_procedures_by_subject(procedures, args.output_path, subjects=subjects, verbose=args.verbose)

break_up_prescriptions_by_subject(prescriptions, args.output_path, subjects=subjects, verbose=args.verbose)


items_to_keep = set(
    [int(itemid) for itemid in dataframe_from_csv(args.itemids_file)['ITEMID'].unique()]) if args.itemids_file else None

for table in args.event_tables:
    read_events_table_and_break_up_by_subject(args.mimic3_path, table, args.output_path, items_to_keep=items_to_keep,
                                              subjects_to_keep=subjects, verbose=args.verbose)
