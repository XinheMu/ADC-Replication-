Welcome to the Model Training codes of ADC
To train the ADC Models, follow the following instructions (note: may require a remote server to run the .sh file)
(1)Prepare a dataset consisting solely of continuous attributes, X, and put it into a csv file 'originalX.csv' in the subdirectory 'Xtraining'
(2)Prepare 30000+ queries generated in the format as queries for the testing set(see the readme.md for testing for query format), and put them into a csv file 'X_trainset.csv' in the subdirectory 'Xtraining'
(3)Prepare the actual cardinality (note: NOT selectivity) of the queries in 'X_trainset.csv' in the form of a 1d numpy array, and put them into a .npy file 'X_real_trainset.npy' in the subdirectory 'Xtraining'
(4)run the command
 bash Train_ADC_onestep.sh X dimension time_min added_layer nan_to
The meaning of the parameters are:
X: Name of your dataset
dimension: dimensionality of your dataset
time_min: early stopping time. 1/320 is a good initial guess, please input this value as a FRACTION rather than DECIMAL
added_layer: add another layer of 150 neurons in the middle of the noise prediction network, recommended 1 for simple datasets and 2 for complex ones
nan_to: The value that you convert rows of missing values to, eg. -1 for the 'power' dataset. Skip if the dataset has no missing values
For my models, that is, run:
bash Train_ADC_onestep.sh forest 10 1/160 2
bash Train_ADC_onestep.sh power 7 1/1280 1 -1
bash Train_ADC_onestep.sh higgs 7 1/1280 1
bash Train_ADC_onestep.sh advantage 5 1/320 1


