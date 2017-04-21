import numpy
import pandas
import matplotlib.pyplot as plt

def load_data_from_csv(filename, row_number_for_column_names):
    return pandas.read_csv(filename, header= row_number_for_column_names)

def convert_numerical_to_categorical(dataframe, column_name, new_column_names, bins, include_minimum=True, include_maximum=True, drop_original_column=True):
    if(include_minimum == True):
        bins = [-100000] + bins
    if(include_maximum == True):
        bins = bins + [100000]
    dataframe['categories'] = pandas.cut(dataframe[column_name], bins, labels=new_column_names)
    categories_dataframe = pandas.get_dummies(dataframe['categories'], prefix= column_name + '=')
    merged_dataframe = dataframe.join(categories_dataframe)
    dataframe = merged_dataframe.drop(['categories'],axis=1)
    if(drop_original_column == True):
        dataframe = merged_dataframe.drop([column_name],axis=1)
    return dataframe

def convert_categorical_to_one_hot_categorical(dataframe, column_name):
    categories_dataframe = pandas.get_dummies(dataframe[column_name], prefix= column_name + '=')
    merged_dataframe = dataframe.join(categories_dataframe)
    dataframe = merged_dataframe.drop([column_name],axis=1)
    return dataframe

def print_unique_value_counts(dataframe, column):
    print dataframe[column].value_counts()

def remove_features(dataframe, feature_columns):
    dataframe = dataframe.drop(feature_columns,axis=1)
    return dataframe

def apply_function_on_column(dataframe, column_name, save_column_name, function):
    dataframe[save_column_name] = dataframe[column_name].apply(function)

def missing_data_seperator(dataframe, column_name):
    null_column_series  = pandas.isnull(dataframe[column_name])
    missing_data_dataframe = dataframe[null_column_series]
    seperated_missing_data_dataframe = dataframe[null_column_series == False]
    return seperated_missing_data_dataframe, missing_data_dataframe

def set_cap_on_numerical_column(dataframe, column_name, max_value):
    dataframe[column_name] = dataframe[column_name].apply(lambda x: x if x<=max_value else max_value)

train_dataframe = load_data_from_csv('train_indessa.csv',0)
train_dataframe.shape

train_dataframe = convert_numerical_to_categorical(train_dataframe,'loan_amnt', ['low','medium','high'], [10000,20000],include_minimum=True, include_maximum=True)
train_dataframe = convert_numerical_to_categorical(train_dataframe,'funded_amnt', ['low','medium','high'], [10000,20000],include_minimum=True, include_maximum=True)
train_dataframe = convert_numerical_to_categorical(train_dataframe,'funded_amnt_inv', ['low','medium','high'], [10000,20000],include_minimum=True, include_maximum=True)

train_dataframe = convert_categorical_to_one_hot_categorical(train_dataframe, 'term')

train_dataframe = remove_features(train_dataframe,['batch_enrolled'])

train_dataframe = convert_categorical_to_one_hot_categorical(train_dataframe, 'grade')
train_dataframe = convert_categorical_to_one_hot_categorical(train_dataframe, 'sub_grade')

train_dataframe = remove_features(train_dataframe,['emp_title','emp_length'])

train_dataframe = convert_categorical_to_one_hot_categorical(train_dataframe, 'home_ownership')
train_dataframe = remove_features(train_dataframe,['home_ownership=_ANY','home_ownership=_MORTGAGE','home_ownership=_NONE','home_ownership=_OTHER','home_ownership=_RENT'])

train_dataframe,missing_annual_inc = missing_data_seperator(train_dataframe, 'annual_inc')
apply_function_on_column(train_dataframe, 'annual_inc', 'annual_inc_log10', numpy.log10)
train_dataframe = convert_numerical_to_categorical(train_dataframe,'annual_inc_log10', ['low','high'], [6.5], include_minimum=True, include_maximum=True, drop_original_column=False)
train_dataframe = remove_features(train_dataframe,['annual_inc', 'annual_inc_log10=_low'])
set_cap_on_numerical_column(train_dataframe, 'annual_inc_log10', 6.5)

train_dataframe = convert_categorical_to_one_hot_categorical(train_dataframe, 'verification_status')

train_dataframe = remove_features(train_dataframe,['pymnt_plan'])

train_dataframe = remove_features(train_dataframe,['desc'])

train_dataframe = remove_features(train_dataframe,['title'])

risky_purpose = ['educational','wedding']
safe_purpose =  ['debt_consolidation','credit_card','home_improvement','other','medical','vacation']
moderate_purpose = ['car','house','major_purchase','moving','renewable_energy','small_business']
apply_function_on_column(train_dataframe, 'purpose', 'purpose=_risky',    lambda x: 1 if x in risky_purpose else 0)
apply_function_on_column(train_dataframe, 'purpose', 'purpose=_moderate', lambda x: 1 if x in moderate_purpose else 0)
apply_function_on_column(train_dataframe, 'purpose', 'purpose=_safe',     lambda x: 1 if x in safe_purpose else 0)
train_dataframe = remove_features(train_dataframe,['purpose'])

train_dataframe = remove_features(train_dataframe,['zip_code'])

safe_state = ['ND','ME','NE','MS','TN','IN']
apply_function_on_column(train_dataframe, 'addr_state', 'addr_state=_safe',    lambda x: 1 if x in safe_state else 0)
train_dataframe = remove_features(train_dataframe,['addr_state'])

apply_function_on_column(train_dataframe, 'dti', 'dti_sqrt', numpy.sqrt)
train_dataframe = convert_numerical_to_categorical(train_dataframe,'dti_sqrt', ['low','high'], [8.0], include_minimum=True, include_maximum=True, drop_original_column=False)
train_dataframe = remove_features(train_dataframe,['dti', 'dti_sqrt=_low'])
set_cap_on_numerical_column(train_dataframe, 'dti_sqrt', 8.0)

train_dataframe = convert_numerical_to_categorical(train_dataframe,'open_acc', ['risky','moderate','safe'], [10,40],include_minimum=True, include_maximum=True)
train_dataframe = remove_features(train_dataframe,['categories'])

train_dataframe = remove_features(train_dataframe,['delinq_2yrs', 'inq_last_6mths', 'mths_since_last_delinq', 'mths_since_last_record'])

train_dataframe = remove_features(train_dataframe,['pub_rec'])

apply_function_on_column(train_dataframe, 'revol_bal', 'revol_bal_log10', lambda x: numpy.log10(x) if x>0 else 0)
train_dataframe = remove_features(train_dataframe,['revol_bal'])

train_dataframe = remove_features(train_dataframe,['revol_util','total_acc','initial_list_status','total_rec_int', 'total_rec_late_fee',	'recoveries',	'collection_recovery_fee',	'collections_12_mths_ex_med', 'mths_since_last_major_derog',	'application_type',	'verification_status_joint',	'last_week_pay',	'acc_now_delinq',	'tot_coll_amt',	'tot_cur_bal',	'total_rev_hi_lim'])

train_dataframe = remove_features(train_dataframe,['member_id'])

def split_target_value(dataframe, column):
    target_value = dataframe[column].copy()
    dataframe    = dataframe.drop([column],axis=1)
    return dataframe, target_value

processed_data, target_value = split_target_value(train_dataframe,'loan_status')

processed_data.shape

from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
logistic_regression_model = LogisticRegression(C=1, max_iter=10000, tol=1e-5, n_jobs=-1)
C = [0.1,1,10]
param_grid = dict(C=C)
grid = GridSearchCV(logistic_regression_model, param_grid, cv=5, scoring='roc_auc')
grid.fit(processed_data, target_value)
grid.grid_scores_
print(grid.best_score_)
print(grid.best_params_)


from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=100,n_jobs=-1)
random_forest.fit(processed_data, target_value)

from sklearn.ensemble import GradientBoostingClassifier
gradient_boosting_classifier = GradientBoostingClassifier(n_estimators=10, learning_rate=0.05, loss='exponential')
gradient_boosting_classifier.fit(processed_data, target_value)
