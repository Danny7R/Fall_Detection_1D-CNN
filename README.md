# Fall Detection using acceleration data with 1D-CNN

The proposed architecture consists of two models, a time-series forecaster which predicts the expected next state of the system by forecasting the next value in each of the time-series, and an anomaly detection model with a binary output indicating the occurrence of a fall. The first model, the time-series forecaster, is conceptually trained only on one class of data, as it will predict the next state under that assumption. Both scenarios, i.e. training on normal data or fall data, have been explored in this project. Before going into the details of the design of the models, let us briefly go over the data preprocessing steps needed for the models.

## Data Preprocessing
After loading the data from the CSV file, the first step is to disregard the missing/corrupted data samples. A portion of the data seems to have contradicting values for “category” and “category_group” features. Any unmatching sample is removed from the dataset. Afterward, a min-max scaler is fitted on the rest of the data so that eventually all values lie within 1 and 0.

To separate the time-series including the occurrence of a fall, samples containing the following words in their corresponding “category_group” are identified using the Pandas library: “front”, “left”, “right”, “back”, and “fall”. Then, all samples with the same ID in the “category_group” column are considered one time-series.

To prepare the data samples for the machine learning algorithms, time-delay embedding is performed on each of the time-series. “tsa.py” contains the functions used for this process. To find the optimal delay and dimension parameters for the delay-embedding, mutual information and false nearest neighbors methods in the TISEAN package are used. After performing the experiments on multiple time-series of the dataset, a delay of 2 proved to be appropriate for the fall containing time-series while the rest indicated 1 as the optimal delay. In order to be consistent, a delay of 1 was chosen for all of the time-series. As for the dimension, the results vary significantly from 8 to 44. Consequently, values from 8 to 60 were explored, with dimensions from 40 and above producing similar results. 50 is the chosen value for this parameter.

## 


_Table 1:
Accuracy | TPR | FPR | FNR
:--:|:--:|:--:|:--:
0.9963 | 0.9963 | 0.0046 | 0.0037

