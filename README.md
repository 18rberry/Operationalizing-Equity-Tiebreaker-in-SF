# DSSG 2021 SFUSD Project

# General Objective

The general objective of this project is to build an equity tiebreaker that can improve the outcome of _focal_ students in the SFUSD school assignment without also benefiting non-focal students.

We will attempt to solve this problem by solving a simpler problem: can we build an equity tiebreaker that can correctly identify the maximum number of _focal_ students while minimizing the number of _non-focal_ students identified incorrectly.

Given the current legislative and social context of the student assignment process in San Francisco, the San Francisco Unified School District (SFUSD) will only use a student's address to assign the equity tiebreaker. These conditions imply that we have to use geographical units to determine each student's eligibility. For this project, we use the _2010 census track blocks_, or _blocks_ as the smallest geographical unit for the assignment of the equity tiebreaker.

A significant limitation of using blocks as a geographical unit is that, in general, a block has both _focal_ and _non-focal_ students. This mixture of student types means that it is impossible to design an equity tiebreaker that perfectly targets a significant number of _focal_ students without benefiting at least some _non-focal_ students. 



# Data

| Block       | Block Group | Geoid10 Group | Demographic Data | FRL and AALPI Data | Student Data |
| ----------- | ----------- | ------------- | ---------------- | ------------------ | ------------ |
| The Block id (also refered to as geoid) is a 15 digit block identificator. This is the smallest geographic unit. | The Block Group id is a 10 digit block identifier that groups multiple blocks. | The Geoid10 Group id was determined by Joseph to group FRL and AALPI Data. It has a one to one correspondence for big block and groups blocks with low counts. | Demographic data at a block level collected from multiple sources. This information is available at a block level. | FRL and AALPI counts at a Geoid10 Group level. This can be used to identify focal students. | Cleaned student data from previous years assignments. This information is available at a student level.|

Important files that might have to be updated when cloning this repository outside of the SOAL cluster:
- `src/d00_utils/file_paths.py`
- `src/d01_data/abstract_data_api.py`
- `src/d01_data/block_data_api.py`
- `src/d01_data/student_data_api.py`
It might also be necessary to generate the pickled version of the block data. An crude example of how to do this can be found in the notebook `20210718-jjl-create-pkl-data.ipynb`.

# Evaluation of Tiebreakers

We are going to consider two methodologies to evaluate different tiebreakers: sample evaluation and counterfactual simulation.

## Sample Evaluation

The sample evaluation consists on comparing the portion of _focal_ students benefited with the equity tiebreaker (true positive rate) with the portion of _non-focal_students_ (false positives).

`TPR = TP / (TP + FN)`

`FNR = FP / (FP + TN)`

## Counterfactual Simulation
The counterfactual simulation consists of using the school assignment simulation engine to evaluate and compare the average performance of _focal_ students in the school assignment process under the proposed equity tiebreaker. This is what is important after all.

To run the counterfactual simulation, we first need to add the new tiebreakers to the student data. Under the current setup of the `dssg_sfusd` repository (this repository) and the `sfusd-project` repository, we have to create and maintain a separate version of the student data that includes a column for each tiebreaker we wish to simulate. The simulation and evaluation of each tiebreaker require three steps:
1. First, we need to update (or create if it doesn't exist) a version of the student data that has a binary column indicating if a student has the tiebreaker. We can do this by using the `SimulationPreprocessing` object located in the `src.d02_intermediate.simulation_preprocessing` module. For the DSSG summer program, we used the notebook `20210730-jjl-preprocess-simulation.ipynb`.
2. Once we update the student data with the new equity tiebreaker we need to run a version of the `sfusd-project` simulation engine that we adapted to run the equity tiebreaker. This version of the simulation engine is in the `dssg-equity-tiebreaker` branch. We can find the necessary path configuration to run this version of the simulator engine in the file`configs/dssg_path_config.yaml` found in this project.
3. Finally, once we have simulated the policy with the corresponding tiebreaker we can use the `SimulationEvaluation` object located in the `src.d06_reporting.simulation_evaluation` module. An example of the use of this object can be found in the `report/` notebooks that evaluate the results.

# Methodologies

## Naive

Order the blocks by how many focal students they have. We consider two orderings, by absolute number of students and by relative number of students (`rate=True`). By relative number of students we mean the proportion of focal students within the block.

## CTIP1

A student is labeled as _focal_ if they live in a block that belongs to the first CTIP quintile. This is the baseline.

## Knapsack Problem

We model the process of selecting or labeling the blocks as a Knapsack Problem. The [Knapsack Problem](https://en.wikipedia.org/wiki/Knapsack_problem) is a problem in combinatorial optimization: Given a set of items, each with a weight and a value, determine which items to include in a collection so that the total weight is less than or equal to a given limit, and the total value is as large as possible.

Under this formulation, the items we are trying to select are the blocks, and the values and weights are the numbers of _focal_  and _non-focal_ students in that block, respectively. The limit on the total weight is our tolerance to the total number of _non-focal_ students that obtain the equity tiebreaker. In other words, this limit is the number of false positives (FP) and is equal to the FPR times the total number of _non-focal_ students (N).

We expect the block selection obtained by this method to yield the maximum TPR for a given FPR.

## Propositions

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
├── report            <- Final analysis as HTML, PDF, LaTeX, etc.
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
