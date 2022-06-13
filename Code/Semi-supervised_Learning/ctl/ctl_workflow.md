# Overview
This markdown file provides a detailed explanation of the steps required to run 
the cluster-then-label (CTL) workflow for semi-supervised learning. Unlike other workflows,
this one is hard-coded at several points, but labels for the required variables at each step
are denoted in all caps (example: `INPUT_DATA`).

The following packages are required for this workflow. Unless specifically indicated,
use the latest version:

  * R:
    * knitr
    * kableExtra)
    * caret
    * plyr
    * dplyr
    * MASS
    * pROC
    * ROCR
    * xgboost
    * haven
    * spatstat.utils
    * imputeTS
    * purrr

  * python
    * pandas
    * numpy
    * tensorflow v1.8 (incompatible with tensorflow 2.0+)
    * hashlib

# Step 1
The file `ctl_step1.R` is an R script that takes as input an imputed training labeled dataset, and 
an imputed unlabeled training dataset. These are referred to in subsequent steps 
as `IMPUTED_LABELED` and `IMPUTED_UNLABELED`, respectively. This step creates a 
train/test validation dataset using the `IMPUTED_LABELED` training labeled dataset, 
performs scaling, and then generates `INPUT_DATA`, which will be used in the next step, and
`MDATA1`, which will be used in subsequent steps.

# Step 2
Step 2 is run using python, and requires tensorflow 1.8 to utilize a builtin KMeans clustering 
library. It takes as input the `INPUT_DATA` file from the previous step. It assumes an optimal k is known, and although it has a default k == 2, other k values
can be assiged instead. It outputs several files that can be used in visualization or debugging,
but the most important one is `MDATA3`, which will be required in step 3 to map cluster identities
to datapoints.

# Step 3
Step 3 is also run using python, and similarly requires tensorflow 1.8. It requires as input 
the `INPUT_DATA`, `MDATA1`, and `MDATA3` files to identify cluster membership of datapoints, 
to identify clusters membership that is above a defined threshold, and to generate an output
file `KLABELS` that will be processed for model generation in step 4. 

# Step 4
Step 4 formats the output from step 3 for use in the XGBoost model building, training, and 
testing. It requires as input the `KLABELS` file from step 3, and the `IMPUTED_LABELED` from
step 1. It outputs `TVAL` and `TTRAIN` files that will be used in step 5.

# Step 5
Step 5 takes as input the clustered and labeled data from the clustering portion of the workflow,
downsamples the labeled data, and then builds an XGBoost model using the train/test validation
data from step 1. After this model is built, the workflow then predicts using the TEST_DATA 
dataset briefly mentioned in step 1, and then outputs the result.
