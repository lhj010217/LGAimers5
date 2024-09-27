# LG Aimers 5th

**Public** : 90th/780 (0.208232)

**Private** : 78th/780 (0.208187)

------------

**Task** : Binary Classification (Normal / AbNormal

**Evaluation metric** : F1-score

**Data** : about 400 features, 30,000 samples. (not uploaded due to copyright restrictions)

------------

**-Data Preprocessing**
1. Removed columns with only one unique value
2. Replaced NaN values with 0
3. Combined train and test datasets (to standardize column names)
4. Applied one-hot encoding
5. Split the combined dataset back into train/test sets
6. Scaled data using Standard Scaler

**-Machine Learning Process**
1. Sampled Normal/Abnormal data
2. Undersampled normal data, oversampled abnormal data
3. Concatenated the sampled data
4. Split the data into train/validation sets (90/10 ratio)
5. Trained the initial model
6. Analyzed feature importance and selected features with an importance score greater than 0.01
7. Performed feature engineering by creating and adding new features
8. Applied selected features to the train/validation/test sets
9. Trained the final model
10. Submitted the outoput

------------

*EDA code not included
