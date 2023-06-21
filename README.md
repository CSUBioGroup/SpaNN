# SpaNN
 
A flexible deep neural network framework for predicting the transcript distribution of the spatial transcriptomics

## Environment
- python 3.9.12
- pytorch 1.12.1
- numpy 1.21.6
- pandas 1.4.4
- scikit-learn 1.0.2


## Data Available
The data used in our research can be downloaded from https://zenodo.org/record/8063157


## Process
1. Unzip data and make dir for saved_model and results.
```
tar -zxvf data.tar.gz
mkdir result
mkdir saved_model
```

2. Run preprocess.py to generate positive samples between spatial transcriptomics.
```
python prerocess.py -c ./configure/osmFISH_Zeisel.yaml
```

3. Run main.py to train and predict the transcript distribution of the spatial transcriptomics.
```
python main.py -c ./configure/osmFISH_Zeisel.yaml
```

## Concat
limin@mail.csu.edu.cn

