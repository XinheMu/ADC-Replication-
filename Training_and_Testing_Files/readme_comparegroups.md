Our train, test, and valid sets, as well as the results for all compare group models, are generated and derived using the open source provided by 
Xiaoying Wang, Changbo Qu, Weiyuan Wu, Jiannan Wang, and Qingqing Zhou. 2021. Are we ready for learned cardinality estimation? Proceedings of the VLDB Endowment (2021)
At website
https://github.com/sfu-db/AreCELearnedYet
We did not use the datasets census and DMV, as they both contain categorical attributes that our model currently cannot handle.
Please consult the original paper and the above link for how to set up their model and run their experiments. 
To convert their generated queries and labels into our preferred format (for comparing our models with the methods they listed; our model can handle queries generated via other methods as long as they are put into a csv file in our preferrred format), use the files "load_labels.py" and "load_queries.py" (note that these two files need to be placed under the working directory for "Are We Ready for Learned Cardinality Estimation" and ran under THEIR working environment as they need to load many of their files)
conda run -n their_environment load_labels.py dataset querytype
conda run -n their_environment load_queries.py dataset querytype
The currently supported values for "dataset" are "forest", "power", "higgs", "advantage"
The currently supported values for "querytype" are "train", "test", "valid"
The generated files will be of the right format and name but in the wrong place, so you would need to move them to the right directory.
(Note: If you wish to convert queries "Are We Ready..." generated for another workload into our format, opening the files load_queries.py and load_labels.py give you a very good idea of how to edit them for the purpose.)
