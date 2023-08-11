# NFL Game Outcome Prediction with Deep Learning

This project demonstrates the process of preparing and augmenting NFL game data for deep learning-based game outcome prediction. It consists of two Python scripts: one for data processing and augmentation, and another for defining and training a deep learning model.

## Table of Contents

- [Description](#description)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)

## Description

### 1 Data Processing Script (`dataprep.py`)

This script processes NFL play-by-play data from a CSV file and performs various calculations and transformations to create a new dataset for analysis. The goal of this script is to prepare the data for further analysis and machine learning tasks related to NFL game outcomes.

The script performs the following main steps:

1. **Data Loading and Trimming**: The script loads NFL play-by-play data from a specified CSV file using the pandas library. It then selects a specific set of columns deemed relevant for further analysis.

2. **Calculate Game Statistics**: The script calculates various game statistics for each team in each game, such as fourth-down conversions, interceptions, yards gained, and more. It creates a new DataFrame containing these calculated statistics.

3. **Team Win Count and Ratios**: The script calculates the total wins per team by year and creates a new feature column representing the ratio of last season's total wins for the home team against the away team.

4. **Data Cleaning and Transformation**: The script performs various data cleaning steps, such as removing rows where the home team is also the away team and removing games that ended in a tie. It also reorders the columns in the DataFrame for consistency.

5. **Feature Engineering**: The script creates a 3-dimensional list to store team statistics and a 2-dimensional list to track recent win counts for each team. It then transforms the home_team and away_team columns into numeric values and generates features based on recent win counts for each team.

6. **Data Normalization**: The script scales the values of the created features in the DataFrame to the range of -1 to 1.

7. **Save Processed Data**: The processed DataFrame is saved both as a CSV file (`data_final.csv`) and a NumPy array file (`data_final.npy`) for further analysis and machine learning tasks.


### 2. Data Augmentation Script (`utils.py`)

Functions within the `utils.py` script are responsible for loading preprocessed NFL game data from the `data_final.npy` file. It performs various data augmentation techniques to enrich the dataset for training a deep learning model. The augmentation techniques employed include:

- Adding random noise to data
- Swapping features between samples with the same label
- Permuting features in data

After augmentation, the data is divided into training and testing sets. The script also generates visualizations to aid in understanding the data relationships. The created visualizations include a correlation matrix plot and scatter plots.

### 3. Deep Learning Model (`models.py`)

The `models.py` script defines a convolutional neural network (CNN) model using TensorFlow and Keras. The model architecture comprises several convolutional layers with batch normalization and activation functions. This is followed by a flatten layer and a dense output layer. The script handles the following:

- Definition of the CNN model using TensorFlow and Keras

### 4. Training (`train.py`)

The `train.py` script defines the training routine. The script handles the following:

- Training the model using the Adamax optimizer and binary cross-entropy loss
- Monitoring training and validation accuracy metrics
- Saving the trained model for future use

## Prerequisites

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- TensorFlow (2.x)

## Installation

1. Clone or download this repository to your local machine.
2. Install the required libraries using pip:

   ```bash
   pip install pandas numpy matplotlib seaborn tensorflow

## Usage

1. Open a terminal and navigate to the directory containing the scripts.
2. Run the data processing and training scripts:

    ```bash
    python -m sports_model.dataprep
    python -m sports_model.train



