# AMD-Diet-UKB

Code for "Development and cross-population evaluation of a machine learning-derived dietary score for age-related macular degeneration" (Wu, Teng, Sheng et al.). Analysis scripts only; the controlled data are not included.

```text
input/      analysis dataset, aMED scores, index summary tables
output/     outputs
```

## Run

```bash
python 01_split_70_30.py
python 02_diet_grouping.py
python 03_diet_cox.py
python 04_ml_sfs.py --wide_csv input/analysis_master_dataset.csv --output_dir output/ml_sfs
python 05_panda_score.py
python 06_amed_score.py
Rscript 07_internal_validation.R
python 08_head_to_head.py
```

## Inputs

- `input/analysis_master_dataset.csv`
- `input/PersonLevel_aMED_CorrectedScores_20260427.csv`
- `output/index_version_panda_amed.csv`
- `output/index_panda_amed_ms_full.csv`

## Requirements

Python: pandas, numpy, scipy, scikit-learn, lightgbm, lifelines, statsmodels, joblib

R: survival, data.table
