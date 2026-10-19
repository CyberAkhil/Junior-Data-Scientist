# Food Inspection Results: Week 1 Analysis

Exploratory analysis and a multiclass classification baseline for the food-inspection workbooks supplied with this project.

## Project contents

- `week1_analysis.py`: profiles the data, evaluates random and chronological holdouts, and summarizes predictions for the unlabeled test rows.
- `Week 2.docx`: a cleaning and preprocessing plan for food-processing and production-sensor data.
- `data/food_inspection_dataset.zip`: the original ZIP supplied for the analysis. It contains `Data_Train.xlsx`, `Data_Test.xlsx`, and `Sample_Submission.xlsx`.
- `Week 1.docx`: the analysis report, results, limitations, and reproducible code appendix.
- `requirements.txt`: Python package requirements.

## Dataset provenance

The supplied files resemble an inspection-outcome dataset for food establishments. Their column names and outcome categories suggest a relationship to municipal food-inspection data, but the ZIP did not include a data dictionary, publisher, extraction record, licence, or source URL. Its fields also contain coded identifiers and years beginning in 2009.

The City of Chicago's public **Food Inspections** portal is a plausible source to investigate, not a verified origin for these exact workbooks:

<https://data.cityofchicago.org/Health-Human-Services/Food-Inspections/4ijn-s7e5>

The portal attributes the dataset to the City of Chicago, directs users to its Terms of Use, and notes that food-inspection procedures changed on July 1, 2018. Verify the original workbook provenance, applicable reuse terms, field definitions, and the target-label mapping before redistributing the data or relying on the results. The target-value-to-class mapping in this analysis is inferred from the sample-submission column order.

This dataset records food-establishment inspection outcomes. It does not contain direct measurements of processed-product composition, sensory quality, or factory process controls.

The Week 2 plan covers general production sensor data and identifies which profiling steps can be applied to the supplied inspection archive. Sensor calibration, time-series cleaning, and process-limit validation require actual sensor readings and metadata, which are not included in the archive.

## Requirements

Python 3.10 or newer is recommended.

```powershell
python -m pip install -r requirements.txt
```

## Run the analysis

From the project root:

```powershell
python week1_analysis.py
```

The script uses `data/food_inspection_dataset.zip` by default. To use a different archive:

```powershell
python week1_analysis.py "C:\path\to\food_inspection_dataset.zip"
```

The script prints data-quality summaries, class balance, random and time-ordered validation metrics, and aggregate test prediction counts. It does not create a submission file.

## Interpretation

The target classes are imbalanced, and the supplied data show a marked change in `PASS` and `PASS(CONDITIONAL)` outcomes beginning in 2017. A model using every supplied field scores substantially better than one restricted to potentially pre-inspection features, but `RiskLevel` and `SectionViolations` may be recorded during or after inspection. Treat this as a leakage risk until feature timing is confirmed.

The chronological validation on 2017-2018 is a more cautious check than a random row split. Pre-inspection features perform only slightly above the historical majority-class baseline on that period. These results do not support using the baseline as an operational food-safety decision tool.
