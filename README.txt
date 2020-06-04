Please run the fils in the following order:
1. data_preprocessing.ipynb 
This is for preprocessing the data collected in the dataset and get the queries.txt and docs.txt for the LSTM model.
2. main_LSTM.ipynb
This is for training the model to learn the semantic meaning behind the sentences and give semantic sentence embedding vector for each users. The distance between two vectors is close if they are both used to present a similar stand.
Note: model_LSTM.py, cells.py are required here.
3. main_GNN.ipynb
This is for training the model with labeled data and give prediction for the unlabelled users. Semi-supervised learning is applied in this part.
Note: model_GNN.py, utils.py are required here.


Environment Setting
Tensorflow version == 1.14

main_LSTM.ipynb is executed under Python 2.7

main_GNN.ipynb and data_preprocessing.ipynb are executed under Python 3.7


The link to access the dataset:
https://drive.google.com/file/d/1238HjdFjLIH6B9y3AVjlnw_5Tjhy0spB/view?usp=sharing
