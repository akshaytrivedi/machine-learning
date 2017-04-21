import numpy
import pandas
import matplotlib.pyplot as plt

def load_data_from_csv(filename, row_number_for_column_names):
    return pandas.read_csv(filename, header= row_number_for_column_names)

train_dataframe = load_data_from_csv('train_indessa.csv',0)
train_dataframe.head()

def print_missing_data_statistics(dataframe):
    print dataframe.isnull().sum()

print_missing_data_statistics(train_dataframe)

def plot_stacked_histogram(dataframe, column_to_plot, condition1, condition2, number_of_bins, labels):
    %matplotlib inline
    from pylab import rcParams
    rcParams['figure.figsize'] = 15, 5
    plt.hist([dataframe[condition1][column_to_plot], dataframe[condition2][column_to_plot]], stacked=True, color = ['g','r'], bins = number_of_bins, label = labels)

def plot_boxplot(dataframe, column_to_plot, condition1, condition2):
    %matplotlib inline
    from pylab import rcParams
    rcParams['figure.figsize'] = 15, 8
    data_to_plot = [dataframe[condition1][column_to_plot], dataframe[condition2][column_to_plot]]
    plt.boxplot(data_to_plot)

plot_stacked_histogram(train_dataframe, 'loan_amnt', train_dataframe['loan_status']==0, train_dataframe['loan_status']==1,50, ['Non-Defaulters','Defaulters'])

plot_boxplot(train_dataframe, 'loan_amnt', train_dataframe['loan_status']==0, train_dataframe['loan_status']==1)

plot_stacked_histogram(train_dataframe, 'funded_amnt', train_dataframe['loan_status']==0, train_dataframe['loan_status']==1,50, ['Non-Defaulters','Defaulters'])

plot_stacked_histogram(train_dataframe, 'funded_amnt_inv', train_dataframe['loan_status']==0, train_dataframe['loan_status']==1,50, ['Non-Defaulters','Defaulters'])

plot_boxplot(train_dataframe, 'funded_amnt_inv', train_dataframe['loan_status']==0, train_dataframe['loan_status']==1)

def split_dataframe_column(dataframe, column_to_split, column_to_save, delimiter=' '):
    delimited_dataframe = dataframe[column_to_split].str.split(delimiter, -1, expand=True)
    dataframe[column_to_save] = delimited_dataframe[0].astype(int)

split_dataframe_column(train_dataframe, 'term', 'term_in_numbers', ' ')

def print_unique_value_counts(dataframe, column):
    print dataframe[column].value_counts()
print_unique_value_counts(train_dataframe,'term_in_numbers')

plot_stacked_histogram(train_dataframe, 'term_in_numbers', train_dataframe['loan_status']==0, train_dataframe['loan_status']==1,50, ['Non-Defaulters','Defaulters'])

print_unique_value_counts(train_dataframe,'annual_inc')

plot_mosaic_plot(train_dataframe, 'sub_grade', 'loan_status')

plot_boxplot(train_dataframe, 'annual_inc', train_dataframe['loan_status']==0, train_dataframe['loan_status']==1)

def missing_data_seperator(dataframe, column_name):
    null_column_series  = pandas.isnull(dataframe[column_name])
    missing_data_dataframe = dataframe[null_column_series]
    seperated_missing_data_dataframe = dataframe[null_column_series == False]
    return seperated_missing_data_dataframe, missing_data_dataframe

train_dataframe,missing_annual_inc = missing_data_seperator(train_dataframe, 'annual_inc')

plot_boxplot(train_dataframe, 'annual_inc', train_dataframe['loan_status']==0, train_dataframe['loan_status']==1)

plot_stacked_histogram(train_dataframe, 'annual_inc', train_dataframe['loan_status']==0, train_dataframe['loan_status']==1,50, ['Non-Defaulters','Defaulters'])

def apply_function_on_column(dataframe, column_name, save_column_name, function):
    dataframe[save_column_name] = dataframe[column_name].apply(function)

apply_function_on_column(train_dataframe, 'annual_inc', 'annual_inc_sqrt', numpy.sqrt)

plot_boxplot(train_dataframe, 'annual_inc_sqrt', train_dataframe['loan_status']==0, train_dataframe['loan_status']==1)

plot_stacked_histogram(train_dataframe, 'annual_inc_sqrt', train_dataframe['loan_status']==0, train_dataframe['loan_status']==1,50, ['Non-Defaulters','Defaulters'])

apply_function_on_column(train_dataframe, 'annual_inc', 'annual_inc_log10', numpy.log10)
plot_boxplot(train_dataframe, 'annual_inc_log10', train_dataframe['loan_status']==0, train_dataframe['loan_status']==1)

print_unique_value_counts(train_dataframe, 'home_ownership')

plot_mosaic_plot(train_dataframe, 'home_ownership', 'loan_status')


print_unique_value_counts(train_dataframe, 'verification_status')

plot_mosaic_plot(train_dataframe, 'verification_status', 'loan_status')

print_unique_value_counts(train_dataframe, 'pymnt_plan')

plot_mosaic_plot(train_dataframe, 'pymnt_plan', 'loan_status')

train_dataframe[train_dataframe['pymnt_plan']=='y']

print_unique_value_counts(train_dataframe, 'desc')





print_unique_value_counts(train_dataframe, 'purpose')

def plot_mosaic_plot(dataframe, column_to_plot, target_column):
    from statsmodels.graphics.mosaicplot import mosaic
    import matplotlib as mpl
    %matplotlib inline
    with mpl.rc_context():
        mpl.rcParams['font.size'] = 16.0
        mpl.rc("figure", figsize=(20,8))
        mosaic(dataframe, [column_to_plot, target_column])

plot_mosaic_plot(train_dataframe, 'purpose', 'loan_status')

def print_mosaic_plot_statistics(dataframe, column_to_plot, target_column, is_ascending=True):
    grouped_dataframe = dataframe.groupby(column_to_plot)[target_column].value_counts().sort_index()
    normal_dataframe = grouped_dataframe.unstack()
    normal_dataframe['ratio'] = normal_dataframe[1]/(normal_dataframe[0] + normal_dataframe[1])
    normal_dataframe.sort_values(by='ratio',inplace=True,ascending=is_ascending)
    print normal_dataframe

print_mosaic_plot_statistics(train_dataframe, 'purpose', 'loan_status')

print_unique_value_counts(train_dataframe, 'addr_state')

print_unique_value_counts(train_dataframe, 'zip_code')

print_mosaic_plot_statistics(train_dataframe, 'zip_code', 'loan_status')

print_mosaic_plot_statistics(train_dataframe, 'addr_state', 'loan_status')

plot_boxplot(train_dataframe, 'dti', train_dataframe['loan_status']==0, train_dataframe['loan_status']==1)

apply_function_on_column(train_dataframe,'dti','dti_sqrt',numpy.sqrt)
plot_boxplot(train_dataframe, 'dti_sqrt', train_dataframe['loan_status']==0, train_dataframe['loan_status']==1)

apply_function_on_column(train_dataframe,'dti','dti_log10',numpy.log10)
plot_boxplot(train_dataframe, 'dti_log10', train_dataframe['loan_status']==0, train_dataframe['loan_status']==1)


print_unique_value_counts(train_dataframe, 'delinq_2yrs')
print_mosaic_plot_statistics(train_dataframe, 'delinq_2yrs', 'loan_status')

print_unique_value_counts(train_dataframe, 'mths_since_last_delinq')


plot_stacked_histogram(train_dataframe, 'revol_bal', train_dataframe['loan_status']==0, train_dataframe['loan_status']==1,50, ['Non-Defaulters','Defaulters'])

plot_boxplot(train_dataframe, 'revol_bal', train_dataframe['loan_status']==0, train_dataframe['loan_status']==1)

apply_function_on_column(train_dataframe,'revol_bal','revol_bal_log10', numpy.log10)
plot_boxplot(train_dataframe, 'revol_bal_log10', train_dataframe['loan_status']==0, train_dataframe['loan_status']==1)


d,m =missing_data_seperator(train_dataframe, 'revol_util')
plot_stacked_histogram(d, 'revol_util', d['loan_status']==0, d['loan_status']==1,50, ['Non-Defaulters','Defaulters'])

plot_boxplot(d, 'revol_util', d['loan_status']==0, d['loan_status']==1)
