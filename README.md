# DSSG 2021 SFUSD Project

# General Objective

The general objective of this project is to build an equity tiebreaker that can correctly identify the maximum number of _focal_ students. It is also important to minimize the number of _non-focal_ students benefited by the equity tiebreaker, i.e., the number of false positives.

Given the current legislative and social context of the student assignment process in San Francisco, the San Francisco Unified School District (SFUSD) will only use a student's address to assign the equity tiebreaker. These conditions imply that we have to use geographical units to determine each student's eligibility. For this project, we use the _2010 census track blocks_, or _blocks_ as the smallest geographical unit for the assignment of the equity tiebreaker.

A significant limitation of using blocks as a geographical unit is that, in general, a block has both _focal_ and _non-focal_ students. This mixture of student types means that it is impossible to design an equity tiebreaker that perfectly targets a significant number of _focal_ students without benefiting at least some _non-focal_ students. 



# Data

| Block       | Block Group | Geoid10 Group | Demographic Data | FRL and AALPI Data | Student Data |
| ----------- | ----------- | ------------- | ---------------- | ------------------ | ------------ |
| The Block id is a 15 digit block identificator. This is the smallest geographic unit. | The Block Group id is a 10 digit block identifier that groups multiple blocks. | The Geoid10 Group id was determined by Joseph to group FRL and AALPI Data. It has a one to one correspondence for big block and groups blocks with low counts. | Demographic data at a block level collected from multiple sources. This information is available at a block level. | FRL and AALPI counts at a Geoid10 Group level. This can be used to identify focal students. | Cleaned student data from previous years assignments. This information is available at a student level.|



# Code setup
__Note__: This code setup is copied from the repo `dssg/hitchhikers-guide`.

The code repository will mirror the data pipeline by creating the corresponding folder structure for the python files.

In addition, there are multiple other files that need to be stored and managed. The community has arrived at a standard setup of the project directories that we will also follow.

Directory structure:

```
├── LICENSE
├── README.md          <- The top-level README for developers using this project.
├── conf
│   ├── base           <- Space for shared configurations like parameters
│   └── local          <- Space for local configurations, usually credentials
│
├── data
│   ├── 01_raw         <- Imutable input data
│   ├── 02_intermediate<- Cleaned version of raw
│   ├── 03_processed   <- The data used for modelling
│   ├── 04_models      <- trained models
│   ├── 05_model_output<- model output
│   └── 06_reporting   <- Reports and input to frontend
│
├── docs               <- Space for Sphinx documentation
│
├── notebooks          <- Jupyter notebooks. Naming convention is date YYYYMMDD (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `20190601-jqp-initial-data-exploration`.
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── results            <- Intermediate analysis as HTML, PDF, LaTeX, etc.
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── .gitignore         <- Avoids uploading data, credentials, outputs, system files etc
│
└── src                <- Source code for use in this project.
    ├── __init__.py    <- Makes src a Python module
    │
    ├── d00_utils      <- Functions used across the project
    │   └── remove_accents.py
    │
    ├── d01_data       <- Scripts to reading and writing data etc
    │   └── load_data.py
    │
    ├── d02_intermediate<- Scripts to transform data from raw to intermediate
    │   └── create_int_payment_data.py
    │
    ├── d03_processing <- Scripts to turn intermediate data into modelling input
    │   └── create_master_table.py
    │
    ├── d04_modelling  <- Scripts to train models and then use trained models to make
    │   │                 predictions
    │   └── train_model.py
    │
    ├── d05_model_evaluation<- Scripts that analyse model performance and model selection
    │   └── calculate_performance_metrics.py
    │    
    ├── d06_reporting  <- Scripts to produce reporting tables
    │   └── create_rpt_payment_summary.py
    │
    └── d06_visualisation<- Scripts to create frequently used plots
        └── visualise_patient_journey.py
```
