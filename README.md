# Offline Q-Function Evaluation for Estimating Soccer Actions Rewards

This repository have the code used for ESE650 Final Project for group 19

## Files
### 0. Install dependences
- Run in terminal pip install -r requirements.txt∂

### 1. prepare_data.py
Before training, is needed to process the raw data:
- Raw data can be downloaded from this drive [link](https://drive.google.com/file/d/1xfwpLU3eR2xzAf-rAJM9gpkL1oPUo7FP/view?usp=sharing)
- Run the prepare_data script to transform and split your data.
- It will output a train and val folder used by train.py to train the model. The processed data is divided in shards to avoid memory overflow.
- Example:
  > python prepare_data.py --input path/to/raw_data --outdir path/to/processed_data --shard-size desired (trajs in each shard) --gamma (discount factor when calculating the returns)

### 2. train.py
Script to train the models:
- Run the train script.
- It will save the models weights to a model/(td0 or mc)/.. folder
- Example:
  > python train.py --shards path/to/processed_data --epochs (max number of epochs) --target (mc for montecarlo, td0 for TD(0)) --trials (numbr of trials on optuna gridsearch) --device (cpu/gpu)

### 3. eval.ipynb
Script to evaluate the model
- Python notebook that loads the model and an event and calculates the return for passes, shot and dribbles
- We provide one trained models in models_example. Also, we provide an example processed shard [here](https://drive.google.com/file/d/1eanI2glIPAxEiEiZp1lH8j4cAYPV3IbK/view?usp=sharing)  which can be downloaded and put in folder data_example to be able to run the eval.ipynb out of the box without processing data or training a model.