# LG Aimers 5th

**Public** : 90th/780 (0.208232)

**Private** : 78th/780 (0.208187)

------------

**Task** : Binary Classification (Normal / AbNormal)

**Evaluation metric** : F1-score

**Data** : about 400 features, 30,000 samples. (not uploaded due to copyright restrictions)

**Timeline** : 2024.08.01 ~ 2024.08.30

------------

**-Data Preprocessing**
1. Remove columns with only one unique value
2. Replace NaN values with 0
3. Combine train and test datasets (to standardize column names)
4. Apply one-hot encoding
5. Split the combined dataset back into train/test sets
6. Scaling data using Standard Scaler

**-Machine Learning Process**
1. Samplling Normal/Abnormal data
   
   => Undersampling normal data, oversampling abnormal data
3. Concatenating the sampled data
4. Split the data into train/validation sets (90/10 ratio)
5. Train the initial model
6. Analyz feature importance and selected features with an importance score greater than 0.01
7. Perform feature engineering by creating and adding new features
8. Apply selected features to the train/validation/test sets
9. Train the final model
10. Submit the outoput

------------

*EDA code not included
