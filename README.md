# MIMIC-III_ICU_Readmission_Analysis
This is the source code for the paper 'Analysis and Prediction of Unplanned Intensive Care Unit Readmission using Recurrent Neural Networks with Long Short-Term Memory'
*[bioRxiv](https://www.biorxiv.org/content/early/2018/08/06/385518)

### Prerequisites

Please follow the original git files from MIMIC-III Benchmark Testing Codes.

```
git clone https://github.com/YerevaNN/mimic3-benchmarks
```

### Step-by-Step
Please follow the steps to get the results:

```
1. python3 scripts/extract_subjects.py [PATH TO MIMIC-III CSVs] data/root/
2. python3 scripts/validate_events.py data/root/
3. python3 scripts/create_readmission.py data/root/
4. python3 scripts/extract_episodes_from_subjects.py data/root/
5. python3 scripts/split_train_and_test.py data/root/
6. python3 scripts/create_readmission_data.py data/root/ data/readmission/
7. python3 mimic3models/split_train_val.py readmission_with_icustay_los
8. cd mimic3models/readmission3/
9. python -u main.py --network ../common_keras_models/lstm.py --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --batch_size 
```

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Citation
If you use this code for your research, please cite our [paper](https://www.biorxiv.org/content/early/2018/08/06/385518/):

```
@article{lin2018analysis,
  title={Analysis and Prediction of Unplanned Intensive Care Unit Readmission using Recurrent Neural Networks with Long Short-Term Memory},
  author={Lin, Yu-Wei and Zhou, Yuqian and Faghri, Faraz and Shaw, Michael J and Campbell, Roy H},
  journal={bioRxiv},
  pages={385518},
  year={2018},
  publisher={Cold Spring Harbor Laboratory}
}

```
