# ics635-higgs-boson

> The goal of this competition is to classify events into two classes: events that produce the exotic Higgs Boson particle, and those that do not. Each event is a simulated particle collision represented by 28 features that describe the trajectories of decay particles. These are realistic simulations of particle collisions occuring in the ATLAS detector at the Large Hadron Collider at CERN, near Geneva, Switzerland.

> An analysis like this was used to discover the Higgs Boson in 2012; a machine learning classifier was trained on simulation and calibration data, then used to analyze real data from the experiment. Improved machine learning methods for this type of data could enable physicists to make new discoveries faster. It is thought that there are additional particles yet to be discovered, so more sensitive machine learning methods could help scientists discover new physics.

Kaggle: https://www.kaggle.com/competitions/higgs-boson-detection-2025/overview

Report: [`./report/report.pdf`](https://github.com/echung32/ics635-higgs-boson/blob/main/reports/report.pdf)

## Project Structure

```
├── README.md          <- The top-level README for developers using this project.
├── data               <- Data files for the project, including training and test datasets.
│   ├── submissions    <- Submissions for the competition.
│   └── models         <- Saved models.
├── scripts            <- Scripts for autogluon (python and slurm)
├── notebooks          <- Jupyter notebooks for exploratory data analysis and experiments.
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
├── requirements.txt   <- The requirements file for reproducing the analysis environment.
└── setup.cfg          <- Configuration file for flake8.
```