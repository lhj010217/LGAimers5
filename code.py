#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install -r requirements.txt')


# In[1]:


import os
from pprint import pprint
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler


# In[2]:


ROOT_DIR = "data"
RANDOM_STATE = 110
train_data_ori = pd.read_csv(os.path.join(ROOT_DIR, "train.csv"))
test_data_ori = pd.read_csv(os.path.join(ROOT_DIR, "test.csv"))
train_data_ori

def preprocess_column(df):
    df = df.replace('OK', 0)
    df_dropped = df.dropna(axis=1, how='all')
    df_dropped = df_dropped.loc[:, df.apply(pd.Series.nunique) > 1]
    df_dropped = df_dropped.fillna(0)
    return df_dropped

train_data_dropped = preprocess_column(train_data_ori)
test_data_dropped = preprocess_column(test_data_ori)
test_data_dropped = test_data_dropped.drop(test_data_dropped.columns[0], axis=1)

def preprocess_data(train_df, test_df, n_components=3000):
    
    # 마지막 열 추출
    last_column_train = train_df.iloc[:, -1]

    # 마지막 열 제외한 데이터 추출
    data_train_without_last_column = train_df.iloc[:, :-1]
    data_test_without_last_column = test_df

    # 두 데이터셋을 결합하여 원핫 인코딩 수행
    combined_data = pd.concat([data_train_without_last_column, data_test_without_last_column], keys=['train', 'test'])
    combined_data_encoded = pd.get_dummies(combined_data)
    combined_data_encoded.fillna(0, inplace=True)
    
    # 다시 train/test 데이터셋으로 분리
    train_encoded = combined_data_encoded.xs('train')
    test_encoded = combined_data_encoded.xs('test')
    
    # 데이터 스케일링 (StandardScaler 사용)
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_encoded)
    test_scaled = scaler.transform(test_encoded)

    # 스케일링된 데이터를 데이터프레임으로 변환
    train_scaled_df = pd.DataFrame(train_scaled, columns=train_encoded.columns)
    test_scaled_df = pd.DataFrame(test_scaled, columns=test_encoded.columns)
    
    # 마지막 열 다시 데이터프레임에 붙이기
    train_processed = pd.concat([train_scaled_df, last_column_train.reset_index(drop=True)], axis=1)
    test_processed = test_scaled_df
    
    return train_processed, test_processed

train_data, test_data = preprocess_data(train_data_dropped, test_data_dropped)

train_data.to_csv("train_processed", index=False)
test_data.to_csv("test_processed", index=False)


# In[ ]:





# In[5]:


train_data = pd.read_csv(os.path.join(ROOT_DIR, "train_processed"))

# "Normal"과 "AbNormal" 샘플링
df_normal = train_data[train_data["target"] == "Normal"]
df_normal = df_normal.sample(n=15000, random_state=RANDOM_STATE)

df_abnormal = train_data[train_data["target"] == "AbNormal"]
df_abnormal = resample(df_abnormal, replace=True, n_samples=12000, random_state=RANDOM_STATE)

# 데이터 병합
df_concat = pd.concat([df_normal, df_abnormal], axis=0).reset_index(drop=True)

# train-validation split
df_train, df_val = train_test_split(
    df_concat,
    test_size=0.1,
    stratify=df_concat["target"],
    random_state=RANDOM_STATE,
)

# CatBoost 모델 초기화
model = CatBoostClassifier(random_state=RANDOM_STATE, verbose=0)

# 피처 리스트 생성 및 데이터 타입 변환
features = []
for col in df_train.columns:
    try:
        df_train[col] = df_train[col].astype(float)
        df_val[col] = df_val[col].astype(float)
        features.append(col)
    except:
        continue

train_x = df_train[features]
train_y = df_train["target"]

val_x = df_val[features]
val_y = df_val["target"]

# 모델 학습
model.fit(train_x, train_y, eval_set=(val_x, val_y))

# 피처 중요도 추출
feature_importances = model.get_feature_importance(Pool(train_x, label=train_y))
feature_importance_dict = dict(zip(features, feature_importances))

# 중요도가 0.01을 넘는 피처 필터링
selected_features = [feat for feat, importance in feature_importance_dict.items() if importance > 0.01]

high_importance_features = [feat for feat, importance in feature_importance_dict.items() if importance > 0.1]
    
def create_engineered_features(df):
    df['Mean Head Coordinate Z Axis'] = (df['HEAD NORMAL COORDINATE Z AXIS(Stage1) Collect Result_Fill1'] + 
                                         df['HEAD NORMAL COORDINATE Z AXIS(Stage2) Collect Result_Fill1'] + 
                                         df['HEAD NORMAL COORDINATE Z AXIS(Stage3) Collect Result_Fill1']) / 3

    df['Pressure Ratio 1_2'] = np.where(df['2nd Pressure Collect Result_AutoClave'] != 0,
                                        df['1st Pressure Collect Result_AutoClave'] / df['2nd Pressure Collect Result_AutoClave'],
                                        np.nan)
    
    df['Pressure Ratio 1_3'] = np.where(df['3rd Pressure Collect Result_AutoClave'] != 0,
                                        df['1st Pressure Collect Result_AutoClave'] / df['3rd Pressure Collect Result_AutoClave'],
                                        np.nan)
    
    df['Pressure Ratio 2_1'] = np.where(df['1st Pressure Collect Result_AutoClave'] != 0,
                                        df['2nd Pressure Collect Result_AutoClave'] / df['1st Pressure Collect Result_AutoClave'],
                                        np.nan)
    
    df['Pressure Ratio 2_3'] = np.where(df['3rd Pressure Collect Result_AutoClave'] != 0,
                                        df['2nd Pressure Collect Result_AutoClave'] / df['3rd Pressure Collect Result_AutoClave'],
                                        np.nan)
    
    df['Pressure Ratio 3_1'] = np.where(df['1st Pressure Collect Result_AutoClave'] != 0,
                                        df['3rd Pressure Collect Result_AutoClave'] / df['1st Pressure Collect Result_AutoClave'],
                                        np.nan)
    
    df['Pressure Ratio 3_2'] = np.where(df['2nd Pressure Collect Result_AutoClave'] != 0,
                                        df['3rd Pressure Collect Result_AutoClave'] / df['2nd Pressure Collect Result_AutoClave'],
                                        np.nan)
    
    df['3rd Pressure Collect Result_AutoClave_Binary'] = df['3rd Pressure Collect Result_AutoClave'].apply(lambda x: 1 if x > 0 else 0)
    
    return df

# 추가된 피처 목록
additional_features = [
    'Mean Head Coordinate Z Axis',
    'Pressure Ratio 1_2',
    'Pressure Ratio 1_3',
    'Pressure Ratio 2_1',
    'Pressure Ratio 2_3',
    'Pressure Ratio 3_1',
    'Pressure Ratio 3_2',
    '3rd Pressure Collect Result_AutoClave_Binary',
]


# 훈련 데이터에 피처 엔지니어링 적용
train_x = create_engineered_features(train_x)

# 검증 데이터에 피처 엔지니어링 적용
val_x = create_engineered_features(val_x)
    
# 테스트 데이터에 피처 엔지니어링 적용
test_data = pd.read_csv(os.path.join(ROOT_DIR, "test_processed"))
df_test_x = create_engineered_features(test_data)

# 선택된 피처와 엔지니어링된 피처를 포함한 데이터 준비
train_x_final = train_x[selected_features + additional_features]
val_x_final = val_x[selected_features + additional_features]
df_test_x_final = df_test_x[selected_features + additional_features]

# 모델 훈련
model_filtered = CatBoostClassifier(random_state=RANDOM_STATE, verbose=2, iterations=1200)
model_filtered.fit(train_x_final, 
                   train_y, 
                   eval_set=(val_x_final, val_y),
                  )

# 예측 수행
test_pred = model_filtered.predict(df_test_x_final)

# 제출 파일 생성
df_sub = pd.read_csv("submission.csv")
df_sub["target"] = test_pred

df_sub.to_csv("submission.csv", index=False)

print("End")


# In[ ]:





# In[ ]:




