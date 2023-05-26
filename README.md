# Identifying Infected Patients Using Semi-Supervised and Transfer Learning 
This repository contains our source code for the work that will be published in the Journal of American Medical Informatics Association (JAMIA). The aim of our project was to identify infection in hospitalized patients with limited gold standard labels from chart-reviewed admissions and a large dataset of admissions with silver standard Sepsis-3 labels using semi-supervised and transfer learning. Several deep learning and non-deep learning algorithms were compared with a baseline algorithm. Semi-supervised learning with non-deep learning methods, as well as transfer learning with deep learning methods did not improve discrimination over a baseline model. However, we found that transfer learning improved calibration.


## Description of Input Data
Train and test data are separate [.csv] files containing patient information collected from EHR data. Predictor variables make up columns of each input data and measurments of each patient within every hour block are stored in a row of the table. Every 25 row of data corresponds to the first 24 hours (hour-0 to hour-24) of patients stay since the admission time. 

## Usage
In this work we have used both R and Python languages. You may need to install keras, keras tuner, and TCN packages in Python, and XGBoost, pROC, and rms packages in R. The code was developed and reviewed using `keras-tuner==1.0.1` and `keras-tcn==3.1.1`. We recommend using these package versions.

**Before you start**

* Make sure data is pre-processed as described in the paper and does not contain any missing value. 
* Before running codes related to deep learning models make an empty `Output` and `Output\Figures` folders within `Transfer_Learning` folder to keep logs of each run of the project. 
* Read the description of each file carefully. Some parts of codes need to be manually editted based on your project and data.

**Non-deep learning models**

There are four models in the [Semi-supervised_Learning](Code/Semi-supervised_Learning) Code folder:
  * The [baseline](Code/Semi-supervised_Learning/nonLSTM_base-model.R) model uses R only, and takes as input an formatted dataset that has been split into training and testing, then builds a model using XGBoost.  For this model, you only need to modify the parameters in the `MODIFY ME` code block at the beginning, then simply run the R script either in RStudio or via commandline.
  * The [sep3](Code/Semi-supervised_Learning/nonLSTM_sep3-model.R) model also uses R only, and takes as input a formatted training dataset, a formatted testing dataset, and a formatted unlabeled dataset. Comments in the code describe how infection labels are assigned using sepsis-3 criteria. For this model, you only need to modify the parameters in the `MODIFY ME` code block at the beginning, then simply run the R script either in RStudio or via commandline.
  * The [self-learning](Semi-supervised_Learning/nonLSTM_sl-model.R) model also uses R only, and takes as input a formatted training dataset, a formatted testing dataset, and a formatted unlabeled dataset. Comments in the code describe how the XGBoost model is built, then used to predict infection status in the unlabeled datapoints, and update these labels, before training and testing a final model. For this model, you only need to modify the parameters in the `MODIFY ME` code block at the beginning, then simply run the R script either in RStudio or via commandline.
  * The [cluster-then-label](Code/Semi-supervised_Learning/ctl) model uses both R and python, and has a separate [README](Code/Semi-supervised_Learning/ctl/ctl_workflow.md) file describing how to run the workflow, including how to configure the clustering steps and the XGBoost model building and testing.

**Deep learning models**

To run codes related to deep learning algorithms, change your current directory in the command line to `Transfer_Learning` directory: `cd Code/Transfer_Learning/`

- To train a [Sepsis-3 model](Code/Transfer_Learning/TL020_train_base.py) run `python TL020_train_base.py`.
- To [fine-tune](Code/Transfer_Learning/TL030_train_tune.py) a pre-trained Sepsis-3 model run `python TL030_train_base.py`. A Sepsis-3 model must be trained first before fine-tuning
- To train a model based on [feature extraction](Code/Transfer_Learning/TL031_feature_extraction.py) using a pre-trained Sepsis-3 model run `python TL031_feature_extraction.py`. 
- After training multiple models, create a single [.csv] file from log_scores of all trained models in which the first column has true labels, and the rest of the file contains prediction probabilities of each model in the gold standard test set. To get AUC, 95% CI of an AUC, and DeLong's test p-value run 
```
Rscript TL052_run_roc_test.R -i /path/to/AllModels_PredScores.csv -p target_label_for_comparison
```


## License
Our code is licensed under a GPL version 3 license (see the license file for detail).

## Citation
Please view our publication on JAMIA. If you find our project useful, please consider citing our work: 

Bashiri et al. Identifying infected patients using semi-supervised and transfer learning. JAMIA 2022. https://academic.oup.com/jamia/article-abstract/29/10/1696/6649188

