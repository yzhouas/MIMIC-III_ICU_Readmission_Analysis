from __future__ import print_function

import os
import argparse
import pandas as pd
import random
random.seed(49297)


parser = argparse.ArgumentParser(description="Create data for readmission prediction task.")
parser.add_argument('root_path', type=str, help="Path to root folder containing train and test sets.")
parser.add_argument('output_path', type=str, help="Directory where the created data should be stored.")
args, _ = parser.parse_known_args()

if not os.path.exists(args.output_path):
    os.makedirs(args.output_path)

total_number_of_patients=0
total_number_of_mortality=0
total_number_of_transfer_back=0
total_number_of_die_in_ward=0
total_number_of_readmit_within_30days=0
total_number_of_die_within_30days=0
total_number_of_readmission=0
total_number_of_not_readmission=0

def process_partition(eps=1e-6):
    number_of_stay=0
    number_of_mortality = 0
    number_of_transfer_back = 0
    number_of_die_in_ward = 0
    number_of_readmit_within_30days = 0
    number_of_die_within_30days = 0
    number_of_readmission = 0
    number_of_not_readmission = 0

    output_dir = os.path.join(args.output_path)
    if (not os.path.exists(output_dir)):
        os.mkdir(output_dir)

    xy_pairs = []
    patients = list(filter(str.isdigit, os.listdir(os.path.join(args.root_path))))
    number_of_patients=len(patients)
    for (patient_index, patient) in enumerate(patients):
        patient_folder = os.path.join(args.root_path, patient)
        patient_ts_files = list(filter(lambda x: x.find("timeseries_readmission.csv") != -1, os.listdir(patient_folder)))
        patient_daty_count=0
        for ts_filename in patient_ts_files:
            with open(os.path.join(patient_folder, ts_filename)) as tsfile:
                lb_filename = ts_filename.replace("_timeseries", "")
                label_df = pd.read_csv(os.path.join(patient_folder, lb_filename))

                # empty label file
                if (label_df.shape[0] == 0):
                    print("\n\t(empty label file)", patient, ts_filename)
                    continue

                los = 24.0 * label_df.iloc[0]['Length of Stay'] # in hours
                if (pd.isnull(los)):
                    print("\n\t(length of stay is missing)", patient, ts_filename)
                    continue

                ts_lines = tsfile.readlines()
                header = ts_lines[0]
                ts_lines = ts_lines[1:]
                event_times = [float(line.split(',')[0]) for line in ts_lines]
                ts_lines = [line for (line, t) in zip(ts_lines, event_times)
                            if t > -eps and t < los + eps]
                event_times = [t for t in event_times
                               if t > -eps and t < los + eps]

                # no measurements in ICU
                if (len(ts_lines) == 0):
                    print("\n\t(no events in ICU) ", patient, ts_filename)
                    continue
                #mortality = int(label_df.iloc[0]["Mortality"])
                readmission= int(label_df.iloc[0]["Readmission"])
                if(readmission==2):
                    print("\n\t(die in ICU)", patient, ts_filename)
                    continue
                elif(readmission==1):
                    number_of_readmission+=1
                else:
                    number_of_not_readmission+=1

                mortality = int(label_df.iloc[0]["Mortality"])
                if (mortality == 1):
                    number_of_mortality += 1

                transferback=int(label_df.iloc[0]['Transfer Back'])
                if (transferback == 1):
                    number_of_transfer_back += 1

                dieinward=int(label_df.iloc[0]['Die in Ward'])
                if (dieinward == 1):
                    number_of_die_in_ward += 1

                readmitwithin30days=int(label_df.iloc[0]['Readmit within 30 days'])
                if (readmitwithin30days == 1):
                    number_of_readmit_within_30days += 1

                diewithin30days=int(label_df.iloc[0]['Die within 30 days'])
                if (diewithin30days == 1):
                    number_of_die_within_30days += 1
                icu_stay=int(label_df.iloc[0]['Icustay'])
                output_ts_filename = patient + "_" + str(icu_stay) +"_"+ts_filename
                with open(os.path.join(output_dir, output_ts_filename), "w") as outfile:
                    outfile.write(header)
                    for line in ts_lines:
                        outfile.write(line)

                xy_pairs.append((output_ts_filename, los,readmission))

                number_of_stay+=1
                patient_daty_count+=1
        if(patient_daty_count==0):
            number_of_patients-=1
        if ((patient_index + 1) % 100 == 0):
            print("\rprocessed {} / {} patients".format(patient_index + 1, len(patients)))

    print("\n", len(xy_pairs))
    random.shuffle(xy_pairs)

    with open(os.path.join(output_dir, "listfile.csv"), "w") as listfile:
        listfile.write('stay,period_length,y_true\n')
        for (x, t, y) in xy_pairs:
            listfile.write("%s,%.6f,%d\n" % (x, t, y))
    return number_of_patients,number_of_stay,number_of_mortality,number_of_transfer_back,number_of_die_in_ward, number_of_readmit_within_30days,number_of_die_within_30days,number_of_readmission, number_of_not_readmission


number_of_patients,number_of_stay,number_of_mortality,number_of_transfer_back,number_of_die_in_ward, number_of_readmit_within_30days,number_of_die_within_30days,number_of_readmission, number_of_not_readmission=process_partition()
print(number_of_patients,number_of_stay,number_of_mortality,number_of_transfer_back,number_of_die_in_ward, number_of_readmit_within_30days,number_of_die_within_30days,number_of_readmission, number_of_not_readmission)
