1. Using the dataset provided, please build a k-NN model for k = 4 that avoids time leakage (details below).


Please refer to HousingPricePredictionKNN.py to find an implementation of 4NN based model that avoids time leakage
* HousingPricePredictionKNN.py: Contains five functions to build the knn model
   * gaussian: provide a gaussian kernel function that accounts for the weight of KNN
   * loadDataset: split dataset into train and test set
   * knn: preprocess the imported data to avoid the time leakage, matrix manipulation and calculation of knn summation, return a list of predicted housing price
   * performanceEval: import lists of actual housing price and predicted housing price , return the relative median absolute error
   * main: import the raw data, clean the raw data, output the measure of performance based on the knn solver


2. What is the performance of the model measured in Median Relative Absolute Error?


Based on a limited computational efficiency that could be further improved, the Relative Median Absolute Error (RMAE) for the current model with a truncated sample (n = 1000) is 0.455 (+/- 0.046).


3. What would be an appropriate methodology to determine the optimal k?


We can use cross-validation to find an optimal number of k value based on the metric of performance. With a varying k value from 1 to 10, the optimal value based on the RMAE score would be k = 2 for this solver.


4. Do you notice any spatial or temporal trends in error?


Since most of the houses in the dataset are located in the geographical location of Kansas and Oklahoma, the dataset is thus with a spatial imbalance problem that can cause spatial error of the knn model I use. This could cause inaccuracy, resulting much larger median relative absolute error of the predictions.




5. How would you improve this model?


There are several ways to improve the model. 
My model currently use a Gaussian weight when calculating knn average. But if similarity between home does not mainly depend on the geographic distance, it is worth checking with cross-validation that if a uniform weight is better than a Gaussian weight.


On the other hand, considering the value of homes could be determined by characteristics other than location and temporality, a logical next step would be to understand the key features that accurately describe the value of a home. For example, I would also introduce a temporal feature with the hypothesis to current dataset that we could check whether the closer closing dates between two houses would result in a more correlated closing price due to shared temporal market conditions.
Then we will check if this hypothesis is valid, according to the contribution of this feature, adjust the weights to improve the model.




6. How would you productionize this model?
Though my implementation uses matrix computation to avoid iterations, it is still computationally expensive. Next step, we need parallelize the knn solver so that the production model will have low-latency prediction capability.


Another way in addition to the first point would be to partition the feature space. If keeping a low-space dimensions of the feature, it is possible to build a k-D tree to query distances between homes as opposed to doing repetitive computations.