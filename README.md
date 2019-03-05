3d-form
==============================

3d-form

Running Experiments
------------

We make use of the sacred python package ([github](https://github.com/IDSIA/sacred), [docs](https://sacred.readthedocs.io/en/latest/index.html)) to manage our experimments.

Experiments are controlled via a json config file. You can view an example at src/configs/config1.json.

Experiments are launched with `python train_vae.py with configs/config1.json`. This will execute the script with the settings inside config1.json and store all outputs in a numbered directory of src/experiments.

Getting Started
------------

Make sure you have Python 3.6.x installed (Tensorflow doesn't support 3.7): https://www.python.org/downloads/

Linux:

* make create_environment
* activate.sh
* make requirements

Windows:

* install git with unix extensions
* install chocolatey from https://chocolatey.org/
* install make with "choco install make"
* make create_environment
* activate.bat
* make requirements

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

Data
------------

Dataset 1: [Thingi10k](https://arxiv.org/pdf/1605.04797.pdf) ([download link](https://drive.google.com/file/d/0B4_KyPW4T9oGRHdMTGZnVDFHLUU/edit) WARNING: it is 9gb _compressed_ and ~32gb _uncompressed_)
