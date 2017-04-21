import numpy
import pandas
import matplotlib.pyplot as plt

def load_data_from_csv(filename, row_number_for_column_names):
    return pandas.read_csv(filename, header= row_number_for_column_names)

train_dataframe = load_data_from_csv('train_indessa.csv',0)

# trying out different things that I learnt from others
# 1) Start by Feature imprtance

# New columns unique values
for col in train_dataframe:
    print(col, ": ", train_dataframe[col].unique())

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score

# Drop irrelevant features and text features
drop_cols = ['member_id', 'batch_enrolled', 'desc', 'title', 'emp_title']
train_dataframe.drop(drop_cols, 1, inplace=True)

train_dataframe['term'] = train_dataframe['term'].str.extract('(\d+)', expand=False).astype(int)
train_dataframe = train_dataframe.fillna("0")
train_dataframe['term'] = train_dataframe['term'].astype(int)

train_dataframe['last_week_pay'] = train_dataframe['last_week_pay'].str.extract('(\d+)', expand=False)
train_dataframe = train_dataframe.fillna("0")
train_dataframe['last_week_pay'] = train_dataframe['last_week_pay'].astype(int)

train_dataframe = train_dataframe.fillna("0")

# Encode Label for Classifier
from sklearn.preprocessing import LabelEncoder

cat_cols = ['grade', 'sub_grade', 'emp_length', 'home_ownership', 'verification_status',
            'pymnt_plan', 'purpose', 'initial_list_status', 'application_type',
            'verification_status_joint', 'zip_code', 'addr_state']
le = {}
for col in cat_cols:
    le[col] = LabelEncoder()
    train_dataframe[col] = le[col].fit_transform(train_dataframe[col])
    le[col].classes_ = numpy.append(le[col].classes_, 'other')

    print('Encoded: ', col)

train_dataframe.shape

def split_target_value(dataframe, column):
    target_value = dataframe[column].copy()
    dataframe    = dataframe.drop([column],axis=1)
    return dataframe, target_value

train_dataframe, target_value = split_target_value(train_dataframe,'loan_status')

rf = RandomForestClassifier(n_estimators=20, n_jobs=-1)
rf.fit(train_dataframe, target_value)
rf.feature_importances_

fi = list(zip(train_dataframe.columns.values, rf.feature_importances_))
fi = sorted(fi, key=lambda x: -x[1])
pandas.DataFrame(fi, columns=["Feature","Importance

# four most important features, performance score = 0.8685
major = ['tot_cur_bal', 'last_week_pay', 'total_rev_hi_lim', 'int_rate']

# not so important features, performance score = 0.7634
minor = ['tot_cur_bal','zip_code', 'addr_state', 'revol_util', 'revol_bal', 'sub_grade', 'annual_inc', 'total_rec_int']

# creating model using only major features
major_train_dataframe = train_dataframe[major]
major_train_dataframe.shape

rf = RandomForestClassifier(n_estimators=25, n_jobs=-1)
rf.fit(major_train_dataframe, target_value)

data_test = pandas.read_csv("test_indessa.csv")
rows = data_test['member_id'].copy()
data_test = data_test[major]

for col in data_test:
    print(col, ": ", data_test[col].unique())

data_test['last_week_pay'] = data_test['last_week_pay'].str.extract('(\d+)', expand=False).astype(float)

data_test = data_test.fillna(0)
pred_test = rf.predict_proba(data_test)
pred_frame = pandas.DataFrame({'member_id': rows, 'loan_status': pred_test[:,1]})
pred_frame.to_csv('learning_submission_v1.csv', index=False, columns=['member_id', 'loan_status'])

pred_frame[pred_frame['member_id']==1000004] = [0.99, 1000004]
pred_frame[pred_frame['member_id']==10006822]
pred_frame['loan_status'] = pred_frame['loan_status'].apply(lambda x: 0.99 if x==1.0 else x)
pred_frame['loan_status'] = pred_frame['loan_status'].apply(lambda x: 0.01 if x==0.0 else x)
