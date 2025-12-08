Welcome to the open source code for “Downsizing Diffusion Models for Cardinality Estimation”

Our open source code contains two parts:

(1)	The Testing_Files directory, used purely for testing the performance of our pre-trained model.

(2)	The Training_and_Testing_Files directory, used for replicating the training process of our model, and then testing their performance.

Our train, test, and valid sets, as well as the results for all compare group models, are generated and derived using the open source provided by 

Xiaoying Wang, Changbo Qu, Weiyuan Wu, Jiannan Wang, and Qingqing Zhou. 2021. Are we ready for learned cardinality estimation? Proceedings of the VLDB Endowment (2021)

At website

https://github.com/sfu-db/AreCELearnedYet

We did not use the datasets census and DMV, as they both contain categorical attributes that our model currently cannot handle.

Please consult the original paper and the above link for how to set up their model and run their experiments. 

Follow the following steps to prepare our repository and environment on your server

1.  Clone the repository

    bash
    
    git clone https://github.com/XinheMu/ADC-Replication-
    
    cd Training_and_Testing_Files
    

3.  Create and activate the Conda environment:

    bash
    
    conda env create -f environment.yml
    
    conda activate my_conda_env
    

Follow these steps to replicate the analysis and results presented in the paper.

## 1.Using Pre-Trained models to directly produce the expeimental results for our workload (or any other workload on our four test datasets)

First, make sure you are under the directory Testing_Files

The current ADC program can be ran in the format

conda run -n environment python ADC_Cardest.py dataset_name ADCversion output_type unit_of_variables dimension Time_min workload_size bayes_source_attribute bayes_called_attribute bayes_assist_attribute nan_to threshold draws

For simplicity, just run

conda run -n my_conda_env ADC_Cardest.py higgs ADC+ qerror "[1e-3,1e-3,1e-3,1e-3,1e-3,1e-3,1e-3]" 7 1/1280 10000 "[1]" "[0]" "[2]"

conda run -n my_conda_env ADC_Cardest.py power ADC+ qerror "[1e-3,1e-3,1e-2,2e-1,1,1,1]" 7 1/1280 10000 "[3]" "[0]" "[6]" -1

conda run -n my_conda_env ADC_Cardest.py forest ADC+ qerror "[1,1,1,1,1,1,1,1,1,1]" 10 1/160 10000

conda run -n my_conda_env ADC_Cardest.py advantage ADC+ qerror "[1,1,1,1,1]" 5 1/320 10000

For a more detailed guide, the model parameters are explained as follows:

dataset_name: The dataset on which to run our experiment. Currently chosen among the values 'forest', 'power', 'higgs', 'advantage'. Note that the dataset 'modulo' is codenamed 'advantage' in our numerical experiments.

ADCversion: Choose among the values 'ADC-','ADC','ADC+'

output_type: Set to 'qerror' for the program to output and display the Q-error; set to 'sel' for the program to calculate the selectivity without calculating th Q-error.

The answer sheet found at location dataset_name+'/'+dataset_name+'_real_test.npy', eg. 'power/power_real_test.npy', is needed for output type 'qerror', but not for 'sel', as the calculation of Q-error require knowledge of the actual selectivity

unit_of_variables: a list enclosed by "" indicating the numerical precision of each attribute, used for preprocessing the query. Eg. unit_of_variables equal 1 for integral attributes, 1e-1 for attributes rounded to 1 digit decimals, 1e-2 for those rounded to 2 digit decimals, etc.

dimension: dimensionality of the dataset

Time_min: early stopping time of the diffusion model

workload_size: Total number of queries to test

bayes_source_attributes; bayes_called_attributes; bayes_assist_attributes: parameters telling the model which attributes we built the Bayesnet on

nan_to: Which number did missing values get converted to, used for the dataset 'power' which contain missing values, and whose missing values we converted to -1 in accrodance with the paper "Are We Ready for Learned Cardinality Estimtion"

threshold (optional): If output_type is set to 'qerror', all queries with an error bigger than threshold will be outputted and their index will be stored to location dataset_name+'/'+dataset_name+'_high_error_list.npy', eg. 'power/power_high_error_list.npy'

draws (optional): Number of draws for predictor-corrector Monte Carlo scheme, default number 25 balances speed with precision according to my tests, but feel free to adjust if you like.

All results are stored in the CSV file dataset_name+'/'+'Statistics_'+dataset_name+'.xlsx', eg. 'power/Statistics_power.xlsx', as a new sheet titled 'QError_'+ADCversion or one titled 'Selectivity_'+ADCversion, eg. "QError_ADC"; "Selectivity_ADC-"

In output mode 'qerror', the five colums are: 'relseldis', the actual selectivity of queries sorted in ascending order; 'relsel', the actual selectivity of queries; 'estsel', the estimated selectivity of queries; 'Q', the q-error for each query; 'SortQ', q-error of all queries sorted in ascending order

In output mode 'sel', the two columns are:'estseldis', the estimated selectivity of queries sorted in ascending order; 'estsel', the estimated selectivity of different queries

Feel free to use new test queries if you like. In that case, to test k new queries on my model, please edit the file dataset_name+'/'+dataset_name+'_testset.csv', eg. 'power/power_testset.csv' to contain 2k rows and 'dimension' columns, with row (2k-2) containing the query lower bounds for each attribute, and row (2k-1) containing the query upper bounds for each atribute.

Note that attributes NOT included in the "WHERE" clause STILL need to appear in the .csv file, in this case, their respective query [lower bound]/[upper bound] is simple the [lower bound of that attribute in the dataset, minus 1]/[upper bound of that attribute in the dataset, plus 1].

Adjust the file dataset_name+'/'+dataset_name+'_real_test.npy' to store the real selectivity of your new queries if you want my program to output Q-error rather than selectivity values while processing them. 

## 2.For training the models, then testing them, which require a remote server

First, make sure you are under the directory Training_and_Testing_Files

For simply replicating the results in my models, please run

bash Train_ADC_onestep.sh forest 10 1/160 2

bash Train_ADC_onestep.sh power 7 1/1280 1 -1

bash Train_ADC_onestep.sh higgs 7 1/1280 1

bash Train_ADC_onestep.sh advantage 5 1/320 1

then run

conda run -n my_conda_env python Train_ADC_Classifier.py forest "[1,1,1,1,1,1,1,1,1,1]" 10 1/160 20000

conda run -n my_conda_env python Train_ADC_Classifier.py power "[1e-3,1e-3,1e-2,2e-1,1,1,1]" 7 1/1280 20000 "[3]" "[0]" "[6]" -1

conda run -n my_conda_env python Train_ADC_Classifier.py higgs "[1e-3,1e-3,1e-3,1e-3,1e-3,1e-3,1e-3]" 7 1/1280 20000 "[1]" "[0]" "[2]"

conda run -n my_conda_env python Train_ADC_Classifier.py advantage "[1,1,1,1,1]" 5 1/320 20000

then run the exact same testing command (that is, run ADC_Cardest.py) using the same commands as before.

For training the workload on a different dataset, named X, please follow the instructions below. Note that X must be made to only contain continuous attributes. The current version of ADC cannot yet handle categorical ones.

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

(5)run the command

conda run -n my_conda_env python Train_ADC_Classifier.py X unit_of_variables dimension Time_min trainset_size bayes_source_attributes bayes_called_attributes bayes_assist_attributes nan_to

To check the bayes_source/called/assist_attributes, look into the directory X for a file named FD3_abc_X.npy. If it doesn't exist, bayesnet was not used and enter the three attributes as "[]", if it does, b is the bayes_source attribute, a the bayes_called_attribute, and c the bayes_assist_attribute

## 3. Getting results for the compare group models, and generating new workloads and query labels in our desired format

Our train, test, and valid sets, as well as the results for all compare group models, are generated and derived using the open source provided by

Xiaoying Wang, Changbo Qu, Weiyuan Wu, Jiannan Wang, and Qingqing Zhou. 2021. Are we ready for learned cardinality estimation? Proceedings of the VLDB Endowment (2021)

At website

https://github.com/sfu-db/AreCELearnedYet

We did not use the datasets census and DMV, as they both contain categorical attributes that our model currently cannot handle.

Please consult the original paper and the above link for how to set up their model and run their experiments.

To convert their generated queries and labels into our preferred format (for comparing our models with the methods they listed; our model can handle queries generated via other methods as long as they are put into a csv file in our preferrred format), use the files "load_labels.py" and "load_queries.py" (note that these two files need to be placed under the working directory for "Are We Ready for Learned Cardinality Estimation" and ran under THEIR working environment, NOT OURS, as the programs need to load many of their files)

conda run -n their_environment load_labels.py dataset querytype

conda run -n their_environment load_queries.py dataset querytype

The currently supported values for "dataset" are "forest", "power", "higgs", "advantage"

The currently supported values for "querytype" are "train", "test", "valid"

The generated files will be of the right format and name but in the wrong place, so you would need to move them to the right directory.

Please put the testing files in the directory “dataset” (eg. forest, power) and put the training queries and labels in the directory “datasettraining” (eg. foresttraining, powertraining), under directory Testomg_Files (for just testing)or Training_and_Testing_Files (for training and testing)
