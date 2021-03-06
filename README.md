# Fall Detection using acceleration data with 1D-CNN

The proposed architecture consists of two models, a time-series forecaster which predicts the expected next state of the system by forecasting the next value in each of the time-series, and an anomaly detection model with a binary output indicating the occurrence of a fall. The first model, the time-series forecaster, is conceptually trained only on one class of data, as it will predict the next state under that assumption. Both scenarios, i.e. training on normal data or fall data, have been explored in this project. Before going into the details of the design of the models, let us briefly go over the data preprocessing steps needed for the models.

## Data Preprocessing

After loading the data from the CSV file, the first step is to disregard the missing/corrupted data samples. A portion of the data seems to have contradicting values for “category” and “category_group” features. Any unmatching sample is removed from the dataset. Afterward, a min-max scaler is fitted on the rest of the data so that eventually all values lie within 1 and 0.

To separate the time-series including the occurrence of a fall, samples containing the following words in their corresponding “category_group” are identified using the Pandas library: “front”, “left”, “right”, “back”, and “fall”. Then, all samples with the same ID in the “category_group” column are considered one time-series.

To prepare the data samples for the machine learning algorithms, time-delay embedding is performed on each of the time-series. “tsa.py” contains the functions used for this process. To find the optimal delay and dimension parameters for the delay-embedding, mutual information and false nearest neighbors methods in the TISEAN package are used. After performing the experiments on multiple time-series of the dataset, a delay of 2 proved to be appropriate for the fall containing time-series while the rest indicated 1 as the optimal delay. In order to be consistent, a delay of 1 was chosen for all of the time-series. As for the dimension, the results vary significantly from 8 to 44. Consequently, values from 8 to 60 were explored, with dimensions from 40 and above producing similar results. 50 is the chosen value for this parameter.

## Architectures and Results

All models explored for the time-series forecasting algorithm were based on 1-dimensional convolutional layers. The simplest architecture included a Conv1D layer followed by the fully connected output layer while the most complicated incorporated multiple residual blocks followed by extra convolutional and fully connected layers. Batch normalization, max pooling, and dropout layers were also incorporated within the layers. Best models illustrated an RMSE of around 0.008 to 0.015 (0.8-1.5% of the amplitude) when trained on either normal or fall data separately. Considering that the standard deviation of all four features was around 0.23 (about 20 times larger than the RMSE) the results of this model seem satisfactory. “forecaster.py” includes the designs of this model.

To detect a fall, the second model then uses the outputs of the forecaster and compares them to the actual observations for the next time step. The explored architectures for this model are random forest, SVM classifier, and multi-layer perceptron which were trained using both normal and fall data, as well as semi-supervised methods including isolation forest and one-class SVM which only require one data class. Random forest was the superior algorithm here, obtaining the highest accuracy in all scenarios. Initially, the overall accuracy obtained on the validation dataset was up to 78.8%, while achieving 100% on the training data. Although a difference between the two is often expected, when different regularization methods fail to make this huge gap smaller we can conclude that the second model needs to be provided with more information as inputs to be able to separate the two classes properly. Subsequently, other than the predicted next state, the means, standard deviations, mins, maxes, 25th percentiles, and 75th percentiles of the 4 features (time-series delay vectors) in the input are also added as inputs to the anomaly detector. Hence, the anomaly detector will compare the actual observations of the next time step with not only the predicted values, but also some statistical properties of the previous values. Table 1 illustrates the final results measured on the validation set (20% of the dataset) and Figure 1 demonstrates the final outline of the model. “anomaly.py” contains the coding of these models.

_Table 1:_
Accuracy | TPR | FPR | FNR
:--:|:--:|:--:|:--:
0.9963 | 0.9963 | 0.0046 | 0.0037


_Figure 1_
