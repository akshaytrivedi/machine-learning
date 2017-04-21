test_dataframe = load_data_from_csv('test_indessa.csv',0)

train_dataframe.shape
test_dataframe.shape

test_dataframe = convert_numerical_to_categorical(test_dataframe,'loan_amnt', ['low','medium','high'], [10000,20000],include_minimum=True, include_maximum=True)
test_dataframe = convert_numerical_to_categorical(test_dataframe,'funded_amnt', ['low','medium','high'], [10000,20000],include_minimum=True, include_maximum=True)
test_dataframe = convert_numerical_to_categorical(test_dataframe,'funded_amnt_inv', ['low','medium','high'], [10000,20000],include_minimum=True, include_maximum=True)

test_dataframe = convert_categorical_to_one_hot_categorical(test_dataframe, 'term')

test_dataframe = remove_features(test_dataframe,['batch_enrolled'])

test_dataframe = convert_categorical_to_one_hot_categorical(test_dataframe, 'grade')
test_dataframe = convert_categorical_to_one_hot_categorical(test_dataframe, 'sub_grade')

test_dataframe = remove_features(test_dataframe,['emp_title','emp_length'])

test_dataframe = convert_categorical_to_one_hot_categorical(test_dataframe, 'home_ownership')

test_dataframe = remove_features(test_dataframe,['home_ownership=_MORTGAGE','home_ownership=_NONE','home_ownership=_OTHER','home_ownership=_RENT'])

#test_dataframe,missing_annual_inc = missing_data_seperator(test_dataframe, 'annual_inc')
apply_function_on_column(test_dataframe, 'annual_inc', 'annual_inc_log10', numpy.log10)
test_dataframe = convert_numerical_to_categorical(test_dataframe,'annual_inc_log10', ['low','high'], [6.5], include_minimum=True, include_maximum=True, drop_original_column=False)
test_dataframe = remove_features(test_dataframe,['annual_inc', 'annual_inc_log10=_low'])
set_cap_on_numerical_column(test_dataframe, 'annual_inc_log10', 6.5)

test_dataframe = convert_categorical_to_one_hot_categorical(test_dataframe, 'verification_status')

test_dataframe = remove_features(test_dataframe,['pymnt_plan'])

test_dataframe = remove_features(test_dataframe,['desc'])

test_dataframe = remove_features(test_dataframe,['title'])

risky_purpose = ['educational','wedding']
safe_purpose =  ['debt_consolidation','credit_card','home_improvement','other','medical','vacation']
moderate_purpose = ['car','house','major_purchase','moving','renewable_energy','small_business']
apply_function_on_column(test_dataframe, 'purpose', 'purpose=_risky',    lambda x: 1 if x in risky_purpose else 0)
apply_function_on_column(test_dataframe, 'purpose', 'purpose=_moderate', lambda x: 1 if x in moderate_purpose else 0)
apply_function_on_column(test_dataframe, 'purpose', 'purpose=_safe',     lambda x: 1 if x in safe_purpose else 0)
test_dataframe = remove_features(test_dataframe,['purpose'])

test_dataframe = remove_features(test_dataframe,['zip_code'])

safe_state = ['ND','ME','NE','MS','TN','IN']
apply_function_on_column(test_dataframe, 'addr_state', 'addr_state=_safe',    lambda x: 1 if x in safe_state else 0)
test_dataframe = remove_features(test_dataframe,['addr_state'])

apply_function_on_column(test_dataframe, 'dti', 'dti_sqrt', numpy.sqrt)
test_dataframe = convert_numerical_to_categorical(test_dataframe,'dti_sqrt', ['low','high'], [8.0], include_minimum=True, include_maximum=True, drop_original_column=False)
test_dataframe = remove_features(test_dataframe,['dti', 'dti_sqrt=_low'])
set_cap_on_numerical_column(test_dataframe, 'dti_sqrt', 8.0)

test_dataframe = convert_numerical_to_categorical(test_dataframe,'open_acc', ['risky','moderate','safe'], [10,40],include_minimum=True, include_maximum=True)
test_dataframe = remove_features(test_dataframe,['categories'])

test_dataframe = remove_features(test_dataframe,['delinq_2yrs', 'inq_last_6mths', 'mths_since_last_delinq', 'mths_since_last_record'])

test_dataframe = remove_features(test_dataframe,['pub_rec'])

apply_function_on_column(test_dataframe, 'revol_bal', 'revol_bal_log10', numpy.log10)
test_dataframe = remove_features(test_dataframe,['revol_bal'])

test_dataframe = remove_features(test_dataframe,['revol_util','total_acc','initial_list_status','total_rec_int', 'total_rec_late_fee',	'recoveries',	'collection_recovery_fee',	'collections_12_mths_ex_med', 'mths_since_last_major_derog',	'application_type',	'verification_status_joint',	'last_week_pay',	'acc_now_delinq',	'tot_coll_amt',	'tot_cur_bal',	'total_rev_hi_lim'])

print_missing_data_statistics(test_dataframe)

test_dataframe, member_id_dataframe = split_target_value(test_dataframe,'member_id')

output_dataframe_logistic_regression = grid.predict_proba(test_dataframe)
output_dataframe_logistic_regression = pandas.DataFrame({'member_id':member_id_dataframe, 'loan_status':output_dataframe_logistic_regression[:,1]})
output_dataframe_logistic_regression = output_dataframe[['member_id','loan_status']]
output_dataframe_logistic_regression.to_csv('output_linear_regression_v1.csv',index=False)


output_dataframe_random_forest = random_forest.predict_proba(test_dataframe)
output_dataframe_random_forest = pandas.DataFrame({'member_id':member_id_dataframe, 'loan_status':output_dataframe_random_forest[:,1]})
output_dataframe_random_forest = output_dataframe[['member_id','loan_status']]
output_dataframe_random_forest.to_csv('output_random_forest_v1.csv',index=False)


output_dataframe_gradient_boosting_classifier = gradient_boosting_classifier.predict_proba(test_dataframe)
output_dataframe_gradient_boosting_classifier = pandas.DataFrame({'member_id':member_id_dataframe, 'loan_status':output_dataframe_gradient_boosting_classifier[:,1]})
output_dataframe_gradient_boosting_classifier = output_dataframe[['member_id','loan_status']]
output_dataframe_gradient_boosting_classifier.to_csv('output_dataframe_gradient_boosting_classifier_v1.csv',index=False)
