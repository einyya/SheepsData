import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdate
import seaborn as sns
import os
import time
import statsmodels.api as sm
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.nonparametric.smoothers_lowess import lowess
import datetime as dt
from datetime import datetime
import time
import requests
from scipy.stats import chi2_contingency
from scipy.stats import ttest_ind
from scipy.stats import f_oneway
import warnings
import shutils
from matplotlib.colors import Normalize


def downlead_file(type):
    # if type == 'TechCare'
    # if type == 'ivri'
    # if type == 'exception'

    if type == 'TechCare':
        data= pd.read_csv(r'C:\Users\e3bom\OneDrive - post.bgu.ac.il\פרויקט גמר ומחקר\חלק יישומי\פרויקט אירופאי\data\Out_Spr_Mer_21.csv')
        data=clean_data(data,DataBase='TechCare')
        # data.rename(columns={'RFID': 'SheepNum'}, inplace=True)
        # data.rename(columns={'Weight': 'weight'}, inplace=True)
        # data.rename(columns={'Date': 'date'}, inplace=True)

        sheep_num_list = data['RFID']
        for sheep_num in sheep_num_list:
            dataNum = data_per_sheep(data, sheep_num, 'TechCare')
            lowess(dataNum, sheep_num, 'daysFS', 'Weight',DataBase='TechCare')

    if type == 'ivri2':
        data= pd.read_csv(r'C:\Users\e3bom\OneDrive - post.bgu.ac.il\פרויקט גמר ומחקר\חלק יישומי\DATA ומודלים\DATA\row_data\Ivri_2.csv')
        data.columns = ['date', 'date-time', 'SheepNum', 'weight', 'ml', 'duration','cross']
        # data['date'] = data['date'].apply(lambda x: datetime.strptime(x, '%d/%m/%Y'))
        data = data.sort_values(by='date')
        data = Herb_data(data)
        sheep_num_list = data['SheepNum'].astype(int)
        for sheep_num in sheep_num_list:
            dataNum = data_per_sheep(data, sheep_num,'ivri')
            lowess(dataNum, sheep_num, 'daysFS', 'weight',DataBase='ivri')
            plot(sheep_num, 'Scatter_lowess', 'daysFS', 'weight', DataBase='ivri')
            lowess(dataNum, sheep_num, 'daysFS', 'ml', exclude_zero=True,DataBase='ivri')
            plot(sheep_num, 'Scatter_lowess', 'daysFS', 'ml', DataBase='ivri')
            plot(sheep_num, 'Scatter_lowess', 'daysFS', 'ml-agr', DataBase='ivri')

            # lowess(dataNum, sheep_num, 'daysFS', 'weight',DataBase='ivri')
            # lowess(dataNum, sheep_num, 'daysFS', 'ml', exclude_zero=True,DataBase='ivri')

    if type == 'ivri':
        # Download the CSV file web
        url = 'https://ews.agri.gov.il/landing/ivri1/db/data.csv'
        response = requests.get(url)
        content = response.content
        decoded_content = content.decode('utf-8')
        # Save the CSV file
        filename = 'data.csv'
        with open(filename, 'w') as f:
            f.write(decoded_content)
        f.close()
        data = pd.read_csv(filename, header=None)  # set header=0 to use the first row as the header
        data = data.drop(data.columns[7], axis=1)
        data.columns = ['date', 'date-time', 'SheepNum', 'weight', 'ml', 'duration', 'cross', 'SensorNum']
        data['date'] = data['date'].apply(lambda x: datetime.strptime(x, '%d/%m/%Y'))
        data = data.sort_values(by='date')
        data = Herb_data(data)
        # sheep_nums = pd.read_csv(r'C:\Users\e3bom\OneDrive - post.bgu.ac.il\פרויקט גמר ומחקר\חלק יישומי\DATA ומודלים\DATA\information_about_data\4_legs_W\C_W4_leg_22-02-23.csv')
        # sheep_num_list = sheep_nums['Tag']
        sheep_num_list = data['SheepNum'].astype(int)
        for sheep_num in sheep_num_list:
            dataNum = data_per_sheep(data, sheep_num,'ivri')
            lowess(dataNum, sheep_num, 'daysFS', 'weight',DataBase='ivri')
            lowess(dataNum, sheep_num, 'daysFS', 'ml', exclude_zero=True,DataBase='ivri')


    if type == 'exception':
        excaption_table= pd.read_csv(r'C:\Users\e3bom\OneDrive - post.bgu.ac.il\פרויקט גמר ומחקר\חלק יישומי\DATA ומודלים\DATA\information_about_data\exceptions\exception_table.csv')
        specific_date_table = excaption_table.loc[:, ['sheepNum', 'specific_date']]
        # specific_date_table = excaption_table.loc[excaption_table['farm'] == 'ivri', ['sheepNum', 'specific_date']]
        sheep_num_list = specific_date_table['sheepNum']
        for sheep_num in sheep_num_list:
            if len(sheep_num)==5:
                if sheep_num=='22674' or sheep_num=='22709':
                    data = pd.read_csv(r'C:\Users\e3bom\OneDrive - post.bgu.ac.il\פרויקט גמר ומחקר\חלק יישומי\DATA ומודלים\DATA\information_about_data\exceptions\row_data\row_data'+sheep_num+'.csv')
                    data['date'] = data['date'].apply(lambda x: datetime.strptime(x, '%d/%m/%Y'))
                else:
                    data = pd.read_csv(r'C:\Users\e3bom\OneDrive - post.bgu.ac.il\פרויקט גמר ומחקר\חלק יישומי\DATA ומודלים\DATA\Analysis_per _sheep\data_per_sheep\data' + sheep_num + '.csv')
                    data['date'] = data['date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
            else:
                data = pd.read_csv(r'C:\Users\e3bom\OneDrive - post.bgu.ac.il\פרויקט גמר ומחקר\חלק יישומי\DATA ומודלים\DATA\TechCare\Analysis_per _sheep\data_per_sheep\data' + sheep_num + '.csv')
                data = data.sort_values(by='Date')

    if type == 'fiona_exception':
        file_path = r'C:\Users\e3bom\OneDrive - post.bgu.ac.il\פרויקט גמר ומחקר\חלק יישומי\DATA ומודלים\DATA\TechCare\fiona\row_data\Firth Trials Master DataSheet 07Mar23 .xlsx'

        excel_file = pd.ExcelFile(file_path)

        healthtreatments = pd.read_excel(excel_file, sheet_name='healthtreatments')
        ewe = pd.read_excel(excel_file, sheet_name='ewe.welfareassessment')
        lamb = pd.read_excel(excel_file, sheet_name='lambs.welfareassessment')
        # Filter rows based on the condition 'treatorhealthissue == 'Lame - sprayed 4 feet with antibiotic spray''
        exception_table = healthtreatments.loc[healthtreatments['treatorhealthissue'] == 'Lame - sprayed 4 feet with antibiotic spray', ['animalid', 'startdate', 'enddate']]
        exception_table['animalid'] = exception_table['animalid'].astype(int)

        # Get the unique sheep numbers
        sheep_num_list = exception_table['animalid'].unique()

        # Iterate over the sheep numbers
        for sheep_num in sheep_num_list:
            specific_date = exception_table.loc[exception_table['animalid'] == sheep_num, ['startdate']].values[0][0]
            specific_date = np.datetime64(specific_date).astype(datetime.datetime).date()
            if sheep_num in ewe['eweid'].values:
                # If the sheep number exists in 'ewe.welfareassessment', extract the columns
                exception_ewe = ewe.loc[ewe['eweid'] == sheep_num, ['date', 'eweid', 'weight']]
                exception_ewe=date_to_numeric_distance(exception_ewe, 'specific_date', specific_date=specific_date)
                x = exception_ewe['date']
                y = exception_ewe['weight']
                plt.plot(exception_ewe['date'], exception_ewe['weight'], color='red', linewidth=2)
                plt.title("ewe "+ str(sheep_num) +" weight ~ date")
                plt.xlabel('date')
                plt.ylabel('weight')
                plt.scatter(x,y)
                plt.savefig(r"C:\Users\e3bom\OneDrive - post.bgu.ac.il\פרויקט גמר ומחקר\חלק יישומי\DATA ומודלים\DATA\TechCare\fiona\Analysis_per _sheep\plot_per sheep\ewe\plot ewe "+ str(sheep_num) +" weight ~ date.png")
                plt.show()
            elif sheep_num in lamb['lambid'].values:
                # If the sheep number exists in 'lambs.welfareassessment', extract the columns
                exception_lamb = lamb.loc[lamb['lambid'] == sheep_num, ['date', 'lambid', 'weight']]
                exception_lamb = date_to_numeric_distance(exception_lamb, 'specific_date', specific_date=specific_date)
                x = exception_lamb['date']
                y = exception_lamb['weight']
                plt.plot(exception_lamb['date'], exception_lamb['weight'], color='red', linewidth=2)
                plt.title("lamb "+ str(sheep_num) +" weight ~ date")
                plt.xlabel('date')
                plt.ylabel('weight')
                plt.scatter(x,y)
                plt.savefig(r"C:\Users\e3bom\OneDrive - post.bgu.ac.il\פרויקט גמר ומחקר\חלק יישומי\DATA ומודלים\DATA\TechCare\fiona\Analysis_per _sheep\plot_per sheep\lamb\plot lamb "+ str(sheep_num) +" weight ~ date.png")
                plt.show()

            else:
                print(f"Sheep number {sheep_num} does not exist in 'ewe.welfareassessment' or 'lambs.welfareassessment'.")




def process_data(type,variable=None,Num=None):
    if Num is not None:
        dataNum = pd.read_csv(r'C:\Users\e3bom\OneDrive - post.bgu.ac.il\פרויקט גמר ומחקר\חלק יישומי\DATA ומודלים\DATA\Analysis_per _sheep\data_per_sheep\data' + Num + '.csv')
    # if type=='bin':
    #     if type == 'date-time':
    if type == 'aggregate':
        if type == 'ml':
            aggregated_ml = dataNum.groupby('daysFS')['ml'].sum().reset_index()
            dataNum.insert(loc=data.columns.get_loc(ml) + 1, column='aggregated_ml', value=aggregated_ml)
            dataNum.to_csv(r'C:\Users\e3bom\OneDrive - post.bgu.ac.il\פרויקט גמר ומחקר\חלק יישומי\DATA ומודלים\DATA\Analysis_per _sheep\data_per_sheep\data' + Num + '.csv')
            return aggregated_data

def Herb_data(data):
    # Drop column 8
    # data = data.drop(data.columns[7], axis=1)
    data = clean_data(data)
    date_to_numeric_distance(data,'first_day')
    dataC = pd.DataFrame(data)
    dataC.to_csv(r'C:\Users\e3bom\OneDrive - post.bgu.ac.il\פרויקט גמר ומחקר\חלק יישומי\DATA ומודלים\DATA\Herb\herd_dataC\Herb_data.csv')
    Lowess_herb(data, 'daysFS', 'weight')
    Lowess_herb(data, 'daysFS', 'ml', exclude_zero=True)
    return data


def Lowess_herb(data, X, Y, exclude_zero=False):
    if exclude_zero == True:
        data = clean_data(data, exclude_zero=True)
    lowessHerb = sm.nonparametric.lowess(data[Y], data[X], frac=0.3)
    lowessHerb_name = f"LowessHerb_{Y}"
    globals()[lowessHerb_name] = lowessHerb
    dataC = pd.DataFrame(lowessHerb)
    column_name = str(Y)
    dataC.columns = ['daysFS', column_name]
    dataC.to_csv(
        r'C:\Users\e3bom\OneDrive - post.bgu.ac.il\פרויקט גמר ומחקר\חלק יישומי\DATA ומודלים\DATA\Herb\lowess_herb\LowessHerb_' + Y + '.csv')
    return lowessHerb

def clean_data(data, exclude_zero=False, DataBase=None):
    if DataBase == 'TechCare':
        missing_cols = data.columns[data.isnull().any()]
        data.dropna(subset=missing_cols, inplace=True)
        data = data[data['Weight'] != 0]
        return data
    if DataBase is None:
        missing_cols = data.columns[data.isnull().any()]
        data.dropna(subset=missing_cols, inplace=True)
        columns=['SheepNum','weight','ml']
        for column in columns:
                data[column] = data[column].astype(int)
        # data['duration'] = data['duration'].apply(lambda x: x if isinstance(x, int) else int(x.split(':')[0]) * 60 + int(x.split(':')[1]))
        # data['duration'] = pd.to_datetime(data['duration'], format='%H:%M')
        # data = data[data['duration'] <= pd.to_datetime('5:00', format='%H:%M').time()]
        if exclude_zero:
            data = data[data['ml'] != 0].copy()  # Create a copy of the DataFrame
        else:
            data = data.copy()
        data = data[data['ml'] <= 5000]
        data = data[data['weight']<= 90]
        return data



# def weight_normalize(data, center_date, specific_date=None):


def date_to_numeric_distance(data, center_date, specific_date=None):
    # if center_date == 'last_day':
    # if center_date == 'specific_date':
    # if center_date == 'first_day':

    if 'date' in data.columns:
        date_col='date'
    if 'Date' in data.columns:
        date_col='Date'
    dates = pd.to_datetime(data[date_col], format='%d/%m/%Y')
    if center_date =='last_day':
        specific_date=dates.max()
    if center_date =='specific_date':
        specific_date = pd.to_datetime(specific_date, format='%d/%m/%Y')
    if center_date == 'first_day':
        specific_date=dates.min()
    date_diffs = (dates - specific_date).dt.days + 1
    if center_date =='last_day':
        if 'daysFE' in data.columns:
            data.drop('daysFE', axis=1, inplace=True)
        data.insert(loc=data.columns.get_loc(date_col) + 1, column='daysFE', value=date_diffs)
    if center_date =='specific_date':
        if 'daysSP' in data.columns:
            data.drop('daysSP', axis=1, inplace=True)
        data.insert(loc=data.columns.get_loc(date_col) + 1, column='daysSP', value=date_diffs)
    if center_date == 'first_day':
        if 'daysFS' in data.columns:
            data.drop('daysFS', axis=1, inplace=True)
        data.insert(loc=data.columns.get_loc(date_col) + 1, column='daysFS', value=date_diffs)
    return data



    # date_diffs_specific_date = (specific_date - dates.min()).days + 1
    # return date_diffs_specific_date

    # Calculate the first date and the difference between each date and the first date in days
    # first_date = dates.min()
    # date_diffs = (dates - first_date).dt.days + 1


def data_per_sheep(data,sheepNum,DataBase):
    Num = sheepNum
    Num = str(Num)
    if DataBase == 'ivri':
        data = data[data['SheepNum'] == sheepNum]
        data=date_to_numeric_distance(data,'first_day')
        data['date'] = pd.to_datetime(data['date'], format='%d/%m/%Y')
        data = data.sort_values(by='date')
        ml_agr = data.groupby('date')['ml'].transform('sum')
        data.insert(loc=data.columns.get_loc('ml') + 1, column='ml-agr', value=ml_agr)
        data.to_csv(r'C:\Users\e3bom\OneDrive - post.bgu.ac.il\פרויקט גמר ומחקר\חלק יישומי\DATA ומודלים\DATA\Analysis_per _sheep\data_per_sheep\data' + Num + '.csv')
    if DataBase=='TechCare':
        data = data[data['RFID'] == sheepNum]
        date_to_numeric_distance(data, 'first_day')
        data.to_csv(r'C:\Users\e3bom\OneDrive - post.bgu.ac.il\פרויקט גמר ומחקר\חלק יישומי\DATA ומודלים\DATA\TechCare\Analysis_per _sheep\data_per_sheep\data' + Num + '.csv')
    return data


def lowess_median(dataNum, Num, X, Y, exclude_zero=False, frac=None):
    if frac is None:
        frac = 0.3
    Num = str(Num)
    if exclude_zero == True:
        dataNum = clean_data(dataNum, exclude_zero=True)
    lowess = sm.nonparametric.lowess(dataNum[Y], dataNum[X], frac=frac)
    lowess = pd.DataFrame(lowess, columns=['daysFS', 'weight'])
    lowess = lowess.drop_duplicates(subset='daysFS', keep='first')
    lowessNum_name = f"lowess_median{Num}_{Y}"
    globals()[lowessNum_name] = lowess
    dataC = pd.DataFrame(lowess)
    column_name = str(Y)
    dataC.columns = ['daysFS', column_name]
    if Y == 'weight':
        dataC.to_csv(
            r'C:\Users\e3bom\OneDrive - post.bgu.ac.il\פרויקט גמר ומחקר\חלק יישומי\DATA ומודלים\DATA\Analysis_per _sheep\lowess_per_sheep\median\weight\lowess_median' + Num + '_' + Y + '.csv')
    if Y == 'ml':
        dataC.to_csv(
            r'C:\Users\e3bom\OneDrive - post.bgu.ac.il\פרויקט גמר ומחקר\חלק יישומי\DATA ומודלים\DATA\Analysis_per _sheep\lowess_per_sheep\median\ml\lowess_median' + Num + '_' + Y + '.csv')

    return lowess

def lowess(dataNum, Num, X, Y, exclude_zero=False, frac=None,DataBase=None,plot=False):
    dataC= pd.DataFrame(columns=['daysFS', Y])
    if frac is None:
        frac = 0.3
    Num = str(Num)
    if exclude_zero == True:
        dataNum = clean_data(dataNum, exclude_zero=True)
    if Y == 'ml':
        lowess_list = ['ml', 'ml-agr']
        # Iterate over the column names in lowess_list
        for Y in lowess_list:
            if Y == 'ml':
                lowess = sm.nonparametric.lowess(dataNum[Y], dataNum[X], frac=frac)
                lowess = pd.DataFrame(lowess, columns=['daysFS', Y])
                lowess = lowess.drop_duplicates(subset='daysFS', keep='first')
                ml_list=lowess[Y]
            if Y == 'ml-agr':
                lowess = sm.nonparametric.lowess(dataNum[Y], dataNum[X], frac=frac)
                lowess = pd.DataFrame(lowess, columns=['daysFS', Y])
                lowess = lowess.drop_duplicates(subset='daysFS', keep='first')
                ml_agrgap_list=lowess[Y]
        dataC = pd.DataFrame({'daysFS': lowess[X], 'ml':ml_list, 'ml-agr': ml_agrgap_list})
        Y='ml'
    if Y == 'weight':
            lowess = sm.nonparametric.lowess(dataNum[Y], dataNum[X], frac=frac)
            lowess = pd.DataFrame(lowess, columns=['daysFS', Y])
            lowess = lowess.drop_duplicates(subset='daysFS', keep='first')
            lowessNum_name = f"lowess_{Num}_{Y}"
            globals()[lowessNum_name] = lowess
            dataC = pd.DataFrame(lowess)
            column_name = str(Y)
            dataC.columns = ['daysFS',column_name]
    if DataBase == 'exception':
            dataC.to_csv(r'C:\Users\e3bom\OneDrive - post.bgu.ac.il\פרויקט גמר ומחקר\חלק יישומי\DATA ומודלים\DATA\Analysis_per _sheep\lowess_per_sheep\lowess_exception\_'+Y+'\lowess' + Num + '_' + Y + '.csv')
    if DataBase =='ivri':
        if Y == 'weight':
            dataC.to_csv(r'C:\Users\e3bom\OneDrive - post.bgu.ac.il\פרויקט גמר ומחקר\חלק יישומי\DATA ומודלים\DATA\Analysis_per _sheep\lowess_per_sheep\weight\lowess' + Num + '_' + Y + '.csv')
        if Y == 'ml':
            dataC.to_csv(r'C:\Users\e3bom\OneDrive - post.bgu.ac.il\פרויקט גמר ומחקר\חלק יישומי\DATA ומודלים\DATA\Analysis_per _sheep\lowess_per_sheep\ml\lowess' + Num + '_' + Y + '.csv')
        if plot:
            if DataBase == 'exception':
                plot(Num, 'Scatter_lowess', X, Y, DataBase='exception')
            if DataBase == 'ivri':
                plot(Num, 'Scatter_lowess', X, Y, DataBase='ivri')
                # plot(Num, PlotType, X, Y, DataBase=None, specific_date=None):

    if DataBase == 'TechCare':
        if Y == 'Weight':
            dataC.to_csv(r'C:\Users\e3bom\OneDrive - post.bgu.ac.il\פרויקט גמר ומחקר\חלק יישומי\DATA ומודלים\DATA\TechCare\Analysis_per _sheep\lowess_per_sheep\weight\lowess' + Num + '_' + Y + '.csv')
        if plot:
            plot(Num, 'Scatter_lowess', X, Y, DataBase='TechCare')
    return lowess

# --------------------------------------------------4 legs weight--------------------------------------------------------
def parameter_exam(specific_date, median=False):
    W4_leg = pd.read_csv(
        r'C:\Users\e3bom\OneDrive - post.bgu.ac.il\פרויקט גמר ומחקר\חלק יישומי\DATA ומודלים\DATA\information_about_data\4_legs_W\C_W4_leg_22-02-23.csv')
    sheep_num_list_per_analysis = []
    W_per_lowess = []
    W_per_lowess_madian = []
    W4_leg_list = []
    initializition = True
    # frac_list = [i / 10 for i in range(1, 11)]
    frac_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    models_results = pd.DataFrame()
    models_results1 = pd.DataFrame()
    sheep_num_list = W4_leg['Tag']
    specific_date = datetime.strptime(specific_date, '%d/%m/%Y')
    new_date_str = datetime.strftime(specific_date, '%Y-%m-%d')
    # Loop over each sheep
    for frac in frac_list:
        for i, sheep_num in enumerate(sheep_num_list):
            sheep_num = str(sheep_num)
            dataNum = pd.read_csv(
                r'C:\Users\e3bom\OneDrive - post.bgu.ac.il\פרויקט גמר ומחקר\חלק יישומי\DATA ומודלים\DATA\Analysis_per _sheep\data_per_sheep\data' + sheep_num + '.csv')
            dataNum['date'] = pd.to_datetime(dataNum['date'])
            dataNum = dataNum[dataNum['date'] <= pd.Timestamp(2023, 2, 22)]
            medians_per_date = dataNum.groupby('date')['weight'].median().reset_index()
            date_to_numeric_distance(medians_per_date)
            medians_per_date.to_csv(
                r'C:\Users\e3bom\OneDrive - post.bgu.ac.il\פרויקט גמר ומחקר\חלק יישומי\DATA ומודלים\DATA\Analysis_per _sheep\data_per_sheep_until\medians_per_date' + sheep_num + '.csv')
            dataNum.to_csv(
                r'C:\Users\e3bom\OneDrive - post.bgu.ac.il\פרויקט גמר ומחקר\חלק יישומי\DATA ומודלים\DATA\Analysis_per _sheep\data_per_sheep_until\data' + sheep_num + '.csv')

            lowess(dataNum, sheep_num, 'daysFS', 'weight', frac=frac)
            lowess_median(medians_per_date, sheep_num, 'daysFS', 'weight', frac=frac)

            try:
                value_daysFS = dataNum.loc[dataNum['date'] == new_date_str, 'daysFS'].values[0]
            except IndexError:
                continue

            lowess_per_sheep = pd.read_csv(
                r'C:\Users\e3bom\OneDrive - post.bgu.ac.il\פרויקט גמר ומחקר\חלק יישומי\DATA ומודלים\DATA\Analysis_per _sheep\lowess_per_sheep\weight\lowess' + sheep_num + '_weight.csv')
            lowess_median_per_sheep = pd.read_csv(
                r'C:\Users\e3bom\OneDrive - post.bgu.ac.il\פרויקט גמר ומחקר\חלק יישומי\DATA ומודלים\DATA\Analysis_per _sheep\lowess_per_sheep\median\weight\lowess_median' + sheep_num + '_weight.csv')
            value_weight = lowess_per_sheep.loc[lowess_per_sheep['daysFS'] == value_daysFS, 'weight'].values[0]
            value_weight1 = \
            lowess_median_per_sheep.loc[lowess_median_per_sheep['daysFS'] == value_daysFS, 'weight'].values[0]
            sheep_num_list_per_analysis.append(sheep_num)
            W_per_lowess.append(value_weight)
            W_per_lowess_madian.append(value_weight1)

            if i == len(sheep_num_list) - 1:
                if initializition:
                    # if len(models_results) == 0:
                    models_results['SheepNum'] = sheep_num_list_per_analysis
                    models_results1['SheepNum'] = sheep_num_list_per_analysis
                    for sheep_num1 in sheep_num_list_per_analysis:
                        sheep_num1 = int(sheep_num1)
                        weight = W4_leg.loc[W4_leg['Tag'] == sheep_num1, 'Weight_4_legs'].values
                        if len(weight) > 0:
                            W4_leg_list.append(weight[0])
                        else:
                            W4_leg_list.append(None)
                models_results['Weight_4_legs'] = W4_leg_list
                models_results1['Weight_4_legs'] = W4_leg_list
                # print(models_results)
                initializition = False
                frac = str(frac)
                models_results[f"{frac}"] = W_per_lowess
                models_results1[f"{frac}"] = W_per_lowess_madian
                W_per_lowess = []
                W_per_lowess_madian = []
    # data_Frac_wight = pd.DataFrame({'SheepNum': sheep_num_list_per_analysis, 'weight': models_results})
    # data_Frac_wight = data_Frac_wight.sort_values(by='SheepNum')
    models_results.to_csv(
        r'C:\Users\e3bom\OneDrive - post.bgu.ac.il\פרויקט גמר ומחקר\חלק יישומי\DATA ומודלים\DATA\parameter examination\lowess\data_Frac_wight.csv')
    models_results1.to_csv(
        r'C:\Users\e3bom\OneDrive - post.bgu.ac.il\פרויקט גמר ומחקר\חלק יישומי\DATA ומודלים\DATA\parameter examination\lowess\data_Frac_lowess_madian_wight.csv')

def gap_cal(type):
    import pandas as pd

    # Read the CSV file
    if type == 'median':
        gap_cal = pd.read_csv(
            r'C:\Users\e3bom\OneDrive - post.bgu.ac.il\פרויקט גמר ומחקר\חלק יישומי\DATA ומודלים\DATA\parameter examination\lowess\data_Frac_lowess_madian_wight.csv')
    else:
        gap_cal = pd.read_csv(
            r'C:\Users\e3bom\OneDrive - post.bgu.ac.il\פרויקט גמר ומחקר\חלק יישומי\DATA ומודלים\DATA\parameter examination\lowess\data_Frac_wight.csv')

    # Check if 'Weight_4_legs' column exists
    if 'Weight_4_legs' not in gap_cal.columns:
        raise ValueError("Column 'Weight_4_legs' not found in the input file")

    # Find the index of 'Weight_4_legs' column
    index_weight_4_legs = gap_cal.columns.get_loc('Weight_4_legs')
    gap_cal.loc['MSE'] = np.nan
    # Calculate the difference between each column from column 2 onwards and the 'Weight_4_legs' column
    for i, col in enumerate(list(gap_cal.columns[index_weight_4_legs + 1:])):
        diff_col = f'{col}_diff'
        gap_cal[diff_col] = gap_cal[col] - gap_cal['Weight_4_legs']
        gap_cal.insert(index_weight_4_legs + 2 * i + 2, diff_col, gap_cal.pop(diff_col))
        mse = np.mean(gap_cal[diff_col] ** 2)
        gap_cal.at['MSE', diff_col] = mse
    if type == 'median':
        gap_cal.to_csv(r'C:\Users\e3bom\OneDrive - post.bgu.ac.il\פרויקט גמר ומחקר\חלק יישומי\DATA ומודלים\DATA\parameter examination\lowess\data_gap_lowess_madian_cal_wight.csv')
    else:
        gap_cal.to_csv(r'C:\Users\e3bom\OneDrive - post.bgu.ac.il\פרויקט גמר ומחקר\חלק יישומי\DATA ומודלים\DATA\parameter examination\lowess\data_gap_cal_wight.csv')


def optimize_frac(estimated_weight, real_weight):
    min_error = float('inf')
    for f in np.arange(0.1, 1.0, 0.1):
        lowessHerb = sm.nonparametric.lowess(estimated_weight, real_weight, frac=f)
        error = sum(abs(lowessHerb[:, 1] - real_weight))
        if error < min_error:
            min_error = error
            best_frac = f
    return best_frac


# --------------------------------------------------cleaning data--------------------------------------------------------

#

# --------------------------------------------------Functions--------------------------------------------------------


def alerts():
    gap_list = []
    num_neg_slope_days_list = []
    sheep_num_list = sheep_nums['SheepNum']

    # Loop over each sheep
    for sheep_num in sheep_num_list:
        # Call Lowess_Num_plot_ and lowess_info functions
        Lowess_Num_plot_(data, 'daysFS', 'ml', sheep_num, plot=False, HerbPlot=False)
        lowess = Lowess_Num_plot_(data, 'daysFS', 'weight', sheep_num, plot=False, HerbPlot=False)

        gap, num_neg_slope_days = lowess_info(lowess, 'weight')

        # Append the results to the corresponding lists
        gap_list.append(gap)
        num_neg_slope_days_list.append(num_neg_slope_days)

    # Create a DataFrame with the results
    data_Gap = pd.DataFrame({'SheepNum': sheep_num_list, 'Gap': gap_list})
    data_NumNegSlope = pd.DataFrame({'SheepNum': sheep_num_list, 'NumNegSlopeDays': num_neg_slope_days_list})
    #
    # Sort the DataFrame by smallest gap and biggest num_neg_slope_days
    data_Gap = data_Gap.sort_values(by='Gap')
    data_NumNegSlope = data_NumNegSlope.sort_values(by='NumNegSlopeDays', ascending=False)
    data_Gap.to_csv(r'C:\Users\e3bom\OneDrive - post.bgu.ac.il\פרויקט גמר ומחקר\חלק יישומי\DATA ומודלים\DATA\alerts\data_Gap.csv')
    data_NumNegSlope.to_csv(r'C:\Users\e3bom\OneDrive - post.bgu.ac.il\פרויקט גמר ומחקר\חלק יישומי\DATA ומודלים\DATA\alerts\data_NumNegSlope.csv')

    # print 3 head results
    data_Gap_first3 = data_Gap.head(3)
    sheep_3_Gap = data_Gap_first3['SheepNum']

    # Loop over each sheep
    for sheep_num in sheep_3_Gap:
        Lowess_Num_plot_(data, 'daysFS', 'weight', sheep_num, plot=True, specific_date=None, HerbPlot=False)

    data_NumNegSlope_first3 = data_NumNegSlope.head(3)
    sheep_3_NumNegSlope = data_NumNegSlope_first3['SheepNum']
    #
    # Loop over each sheep
    for sheep_NumNegSlope in sheep_3_NumNegSlope:
        Lowess_Num_plot_(data, 'daysFS', 'weight', sheep_NumNegSlope, plot=True, specific_date=None, HerbPlot=False)
    return data_Gap, data_NumNegSlope


def lowess_info(lowess, Y):
    i = 0
    gap = lowess.iloc[-1][Y] - lowess.iloc[- 2][Y]
    print(gap)
    if gap >= 0:
        num_neg_slope_days = 0
        return gap, num_neg_slope_days
    else:
        while True:
            gap = lowess.iloc[-1 - i][Y] - lowess.iloc[-i - 2][Y]
            if gap >= 0:
                break
            i += 1
            if i >= len(lowess):
                break

    gap = lowess.iloc[-1][Y] - lowess.iloc[- 2][Y]
    num_neg_slope_days = i
    return gap, num_neg_slope_days


def filter_lowess(lowess, lowessHerb_ml):
    # Find rows in lowessHerb_ml that have matching values in first column of lowess
    matching_rows = np.isin(lowessHerb_ml[:, 0], lowess[:, 0])
    # Create new array with only the matching rows
    new_lowessHerb_ml = lowessHerb_ml[matching_rows]
    return new_lowessHerb_ml


def custom_date_parser(date_str):
    return pd.datetime.strptime(date_str, '%d/%m/%Y')

def plot(Num,PlotType, X, Y,DataBase = None ,specific_date=None):
    Num=str(Num)
    if DataBase == 'exception':
        dataNum = pd.read_csv(r'C:\Users\e3bom\OneDrive - post.bgu.ac.il\פרויקט גמר ומחקר\חלק יישומי\DATA ומודלים\DATA\information_about_data\exceptions\_'+Num+'\data_'+Num+'_DateCenter.csv')
        if Y == 'ml':
            dataNum=clean_data(dataNum, exclude_zero=True)
    if DataBase == 'ivri':
        dataNum = pd.read_csv(r'C:\Users\e3bom\OneDrive - post.bgu.ac.il\פרויקט גמר ומחקר\חלק יישומי\DATA ומודלים\DATA\Analysis_per _sheep\data_per_sheep\data'+Num+'.csv')
        if Y == 'ml':
            dataNum=clean_data(dataNum, exclude_zero=True)
    if DataBase == 'TechCare':
        dataNum = pd.read_csv(r'C:\Users\e3bom\OneDrive - post.bgu.ac.il\פרויקט גמר ומחקר\חלק יישומי\DATA ומודלים\DATA\TechCare\Analysis_per _sheep\data_per_sheep\data' + Num + '.csv')
        if Y == 'ml':
            dataNum=clean_data(dataNum, exclude_zero=True,DataBase ='TechCare')
    if X == 'date':
        plt.xticks(rotation=90)
        plt.xticks(fontsize=6)
    plt.title("Sheep" + str(Num) + " " + Y + " ~ " + X)
    if PlotType == 'hist':
         plt.hist(dataNum[Y], bins=10, edgecolor='black')
         plt.title("Histogram" + str(Num) + " " + Y + " ~ " + X )
         plt.xlabel(X)
         plt.ylabel("Frequency")
         plt.grid(True)
         if DataBase == 'exception' or DataBase == 'ivri':
            plt.savefig(r"C:\Users\e3bom\OneDrive - post.bgu.ac.il\פרויקט גמר ומחקר\חלק יישומי\DATA ומודלים\DATA\Analysis_per _sheep\plot_per sheep\_hist\_date-time\histogram"+ Num + ".png")
         if DataBase == 'TechCare':
             plt.savefig(r"C:\Users\e3bom\OneDrive - post.bgu.ac.il\פרויקט גמר ומחקר\חלק יישומי\DATA ומודלים\DATA\TechCare\Analysis_per _sheep\plot_per sheep\_hist\_date-time\histogram" + Num + ".png")
         plt.show()
         return
    if DataBase == 'exception' or DataBase == 'ivri':
        if PlotType=='Scatter' or PlotType =='Scatter_lowess':
            x = dataNum[X]
            y = dataNum[Y]
        if PlotType == 'lowess' or PlotType =='Scatter_lowess' or PlotType == 'lowess_less_first' or PlotType == 'lowess_division_first':
            if Y == 'weight':
                lowess = pd.read_csv(r'C:\Users\e3bom\OneDrive - post.bgu.ac.il\פרויקט גמר ומחקר\חלק יישומי\DATA ומודלים\DATA\Analysis_per _sheep\lowess_per_sheep\weight\lowess' + Num + '_' + Y + '.csv')
            if Y == 'ml':
                lowess = pd.read_csv(r'C:\Users\e3bom\OneDrive - post.bgu.ac.il\פרויקט גמר ומחקר\חלק יישומי\DATA ומודלים\DATA\Analysis_per _sheep\lowess_per_sheep\ml\lowess' + Num + '_' + Y + '.csv')
            if PlotType == 'lowess_less_first':
                first_value = lowess[Y][0]
                normalized_y = lowess[Y] - first_value
                x = lowess[X]
                y = normalized_y
                data = pd.DataFrame({'daysFS': x, 'weight': y})
                data.to_csv(r'C:\Users\e3bom\OneDrive - post.bgu.ac.il\פרויקט גמר ומחקר\חלק יישומי\DATA ומודלים\DATA\Analysis_per _sheep\lowess_per_sheep\normalize\lowess_less_first\lowess' + Num + '_' + Y + '.csv')
                plt.plot(x, y, color='blue', linewidth=2, label='Normalized')
                plt.xlabel(X)
                plt.ylabel(Y)
                plt.legend()
                plt.savefig(r"C:\Users\e3bom\OneDrive - post.bgu.ac.il\פרויקט גמר ומחקר\חלק יישומי\DATA ומודלים\DATA\Analysis_per _sheep\plot_per sheep\normalize\lowess_less_first\plot" + str(Num) + Y + "~ " + X + ".png")
                plt.show()
                return
            if PlotType == 'lowess_division_first':
                first_value = lowess[Y][0]
                normalized_y = lowess[Y] / first_value
                x = lowess[X]
                y = normalized_y
                plt.plot(x, y, color='blue', linewidth=2, label='Normalized')
                plt.xlabel(X)
                plt.ylabel(Y)
                plt.legend()
                data = pd.DataFrame({'daysFS': x, 'weight': y})
                data.to_csv(r'C:\Users\e3bom\OneDrive - post.bgu.ac.il\פרויקט גמר ומחקר\חלק יישומי\DATA ומודלים\DATA\Analysis_per _sheep\lowess_per_sheep\normalize\lowess_division_first\lowess' + Num + '_' + Y + '.csv')
                plt.savefig(r"C:\Users\e3bom\OneDrive - post.bgu.ac.il\פרויקט גמר ומחקר\חלק יישומי\DATA ומודלים\DATA\Analysis_per _sheep\plot_per sheep\normalize\lowess_division_first\plot" + str(Num) + Y + "~ " + X + ".png")
                plt.show()
                return

    if PlotType == 'lowess':
        x = lowess[X]
        y = lowess[Y]
        if specific_date is not None:
            specific_date = datetime.strptime(specific_date, '%d/%m/%Y')
            new_date_str = datetime.strftime(specific_date, '%Y-%m-%d')
            value = dataNum.loc[dataNum['date'] == new_date_str, 'daysFS'].values[0]
            plt.axvline(x=value, color='b', linestyle='--')
        plt.xlabel(X)
        plt.ylabel(Y)
        plt.scatter(x,y)
        plt.savefig(r"C:\Users\e3bom\OneDrive - post.bgu.ac.il\פרויקט גמר ומחקר\חלק יישומי\DATA ומודלים\DATA\Analysis_per _sheep\plot_per sheep\_"+ PlotType +"\_"+ Y +"\plot" + str(Num) + Y + "~ " + X + ".png")
        plt.show()
        return
    if PlotType =='Scatter_lowess':
        # specific_date = datetime.strptime(specific_date, '%d/%m/%Y')
        # new_date_str = datetime.strftime(specific_date, '%Y-%m-%d')
        # value = dataNum.loc[dataNum['date'] == new_date_str, 'daysFS'].values[0]
        if Num=='22725':
            first_value = lowess[X][0]
            normalized_y = lowess[Y] / first_value
            x = lowess[X]
            y = normalized_y
        data_gap = pd.read_csv(r'C:\Users\e3bom\OneDrive - post.bgu.ac.il\פרויקט גמר ומחקר\חלק יישומי\DATA ומודלים\DATA\parameter examination\lowess\data_gap_cal_wight.csv')
        data_gap['SheepNum'] = data_gap['SheepNum'].astype(str).str.rstrip('0').str.rstrip('.')
        gap = data_gap.loc[data_gap['SheepNum'] == Num, 'abs(0.3_diff)/Weight_4_legs']
        gap = gap.iloc[0]
        plt.axvline(x=53, color='brown', linestyle='--', label='gap' + str(gap))
        plt.plot(lowess[X], lowess[Y], color='blue', linewidth=2)
        plt.xlabel(X)
        plt.ylabel(Y)
        plt.scatter(x,y)
        plt.legend()
        plt.savefig(r"C:\Users\e3bom\OneDrive - post.bgu.ac.il\פרויקט גמר ומחקר\חלק יישומי\DATA ומודלים\DATA\Analysis_per _sheep\plot_per sheep\groups\ivri1-1.1.23\plot" + str(Num) + Y + "~ " + X + ".png")
        plt.show()

    if DataBase == 'TechCare':

        if PlotType == 'lowess' or PlotType =='Scatter_lowess':
            if Y == 'Weight':
                lowess = pd.read_csv(r'C:\Users\e3bom\OneDrive - post.bgu.ac.il\פרויקט גמר ומחקר\חלק יישומי\DATA ומודלים\DATA\TechCare\Analysis_per _sheep\lowess_per_sheep\weight\lowess' + Num + '_' + Y + '.csv')
            plt.plot(lowess[X], lowess[Y], color='red', linewidth=2)
        if specific_date is not None:
            specific_date = datetime.strptime(specific_date, '%d/%m/%Y')
            new_date_str = datetime.strftime(specific_date, '%Y-%m-%d')
            value = dataNum.loc[dataNum['date'] == new_date_str, 'daysFS'].values[0]
            plt.axvline(x=value, color='b', linestyle='--')
        plt.xlabel(X)
        plt.ylabel(Y)
        plt.scatter(x,y)
        plt.savefig(r"C:\Users\e3bom\OneDrive - post.bgu.ac.il\פרויקט גמר ומחקר\חלק יישומי\DATA ומודלים\DATA\TechCare\Analysis_per _sheep\plot_per sheep\_"+ PlotType +"\_"+ Y +"\plot" + str(Num) + Y + "~ " + X + ".png")
        plt.show()

    if DataBase is None:
        if PlotType == 'multi_exception':
            # exception_table = pd.read_csv(r'C:\Users\e3bom\OneDrive - post.bgu.ac.il\פרויקט גמר ומחקר\חלק יישומי\DATA ומודלים\DATA\information_about_data\exceptions\exception_table.csv')
            # sheep_num_list = exception_table.loc[exception_table['strees_type'] == 'death', 'sheepNum']
            # sheep_num_list = exception_table.loc[(exception_table['strees_type'] == 'death') & (exception_table['farm'] == 'ivri'), 'sheepNum']
            # sheep_num_list = exception_table.loc[(exception_table['strees_type'] == 'tumor') & (exception_table['farm'] == 'ivri'), 'sheepNum']
            # sheep_num_list = exception_table['sheepNum']
            sheep_num_list = ['432','437','450']
            # Define a colormap for assigning colors to each sheep
            colormap = plt.cm.get_cmap('viridis', len(sheep_num_list))

            for i, sheep_num in enumerate(sheep_num_list):
                if Y == 'weight':
                    if len(sheep_num)==5:
                        lowess = pd.read_csv(r'C:\Users\e3bom\OneDrive - post.bgu.ac.il\פרויקט גמר ומחקר\חלק יישומי\DATA ומודלים\DATA\Analysis_per _sheep\lowess_per_sheep\weight\lowess' + sheep_num + '_' + Y + '.csv')
                    else:
                        lowess = pd.read_csv(r'C:\Users\e3bom\OneDrive - post.bgu.ac.il\פרויקט גמר ומחקר\חלק יישומי\DATA ומודלים\DATA\Analysis_per _sheep\lowess_per_sheep\weight\lowess' + sheep_num + '_' + Y + '.csv')
                        Y = 'weight'
                # Assign a color based on the colormap
                color = colormap(i)
                plt.plot(lowess[X], lowess[Y], color=color, linewidth=2, label=f'Sheep {sheep_num}')
            # Set labels and title for the plot
            plt.xlabel(X)
            plt.ylabel(Y)
            plt.title('Lowess Curves for the Comparison')
            plt.legend()
            plt.savefig(r"C:\Users\e3bom\OneDrive - post.bgu.ac.il\פרויקט גמר ומחקר\חלק יישומי\DATA ומודלים\DATA\Analysis_per _sheep\plot_per sheep\_multi_lowess\Comparison "+Y+".png")
            plt.show()
            return
        if PlotType == 'multi_exception_division_first':
            data = pd.read_csv(r'C:\Users\e3bom\OneDrive - post.bgu.ac.il\פרויקט גמר ומחקר\חלק יישומי\DATA ומודלים\DATA\parameter examination\lowess\data_gap_cal_wight_onlyDiff.csv')
            data = data.loc[data['binary'] == 1, ['SheepNum']].astype(int)
            # Define a colormap for assigning colors to each sheep
            sheep_num_list = data['SheepNum']
            colormap = plt.cm.get_cmap('viridis', len(sheep_num_list))
            for i, sheep_num in enumerate(sheep_num_list):
                lowess = pd.read_csv(r'C:\Users\e3bom\OneDrive - post.bgu.ac.il\פרויקט גמר ומחקר\חלק יישומי\DATA ומודלים\DATA\Analysis_per _sheep\lowess_per_sheep\normalize\lowess_division_first\lowess' + str(sheep_num) + '_' + Y + '.csv')
                color = colormap(i)
                plt.plot(lowess[X], lowess[Y], color=color, linewidth=2, label=f'Sheep {sheep_num}')
        # Set labels and title for the plot
            plt.xlabel(X)
            plt.ylabel(Y)
            plt.title('Lowess Curves for the Comparison_division_first')
            plt.legend()
            plt.savefig(r"C:\Users\e3bom\OneDrive - post.bgu.ac.il\פרויקט גמר ומחקר\חלק יישומי\DATA ומודלים\DATA\Analysis_per _sheep\plot_per sheep\_multi_lowess\Comparison_division_first"+Y+".png")
            plt.show()
            return
    if PlotType == 'multi_exception_less_first':
        data = pd.read_csv(r'C:\Users\e3bom\OneDrive - post.bgu.ac.il\פרויקט גמר ומחקר\חלק יישומי\DATA ומודלים\DATA\parameter examination\lowess\data_gap_cal_wight_onlyDiff.csv')
        data = data.loc[data['binary'] == 1, ['SheepNum']].astype(int)
        # Define a colormap for assigning colors to each sheep
        sheep_num_list=data['SheepNum']
        colormap = plt.cm.get_cmap('viridis', len(sheep_num_list))

        for i, sheep_num in enumerate(sheep_num_list):
            lowess = pd.read_csv(r'C:\Users\e3bom\OneDrive - post.bgu.ac.il\פרויקט גמר ומחקר\חלק יישומי\DATA ומודלים\DATA\Analysis_per _sheep\lowess_per_sheep\normalize\lowess_less_first\lowess' + str(sheep_num) + '_' + Y + '.csv')
            color = colormap(i)
            plt.plot(lowess[X], lowess[Y], color=color, linewidth=2, label=f'Sheep {sheep_num}')
            for i, sheep_num in enumerate(sheep_num_list):
                lowess = pd.read_csv(r'C:\Users\e3bom\OneDrive - post.bgu.ac.il\פרויקט גמר ומחקר\חלק יישומי\DATA ומודלים\DATA\Analysis_per _sheep\lowess_per_sheep\normalize\lowess_less_first\lowess' + str(sheep_num) + '_' + Y + '.csv')
                color = red
                plt.plot(lowess[X], lowess[Y], color=color, linewidth=2, label=f'Sheep {sheep_num}')
            # Set labels and title for the plot
            plt.xlabel(X)
            plt.ylabel(Y)
            plt.title('Lowess Curves for the Comparison_less_first_with_exception')
            plt.legend()
            plt.savefig(r"C:\Users\e3bom\OneDrive - post.bgu.ac.il\פרויקט גמר ומחקר\חלק יישומי\DATA ומודלים\DATA\Analysis_per _sheep\plot_per sheep\_multi_lowess\Comparison_less_first" + Y + ".png")
            plt.show()
            return
    if PlotType == 'multi_exception_less_first_with_exception':
            # data = pd.read_csv(r'C:\Users\e3bom\OneDrive - post.bgu.ac.il\פרויקט גמר ומחקר\חלק יישומי\DATA ומודלים\DATA\parameter examination\lowess\data_gap_cal_wight_onlyDiff.csv')
            # sheep_num_list = data.loc[data['binary'] == 1, ['SheepNum']].astype(int)
            # Define a colormap for assigning colors to each sheep
            data=pd.read_csv(r'C:\Users\e3bom\OneDrive - post.bgu.ac.il\פרויקט גמר ומחקר\חלק יישומי\DATA ומודלים\DATA\information_about_data\4_legs_W\C_W4_leg_22-02-23.csv')
            sheep_num_list = data.loc[(data['Entre_Date'] == '01/01/2023') & (data['type'] == 'ok'), 'Tag'].astype(int)
            for i, sheep_num in enumerate(sheep_num_list):
                lowess = pd.read_csv(r'C:\Users\e3bom\OneDrive - post.bgu.ac.il\פרויקט גמר ומחקר\חלק יישומי\DATA ומודלים\DATA\Analysis_per _sheep\lowess_per_sheep\normalize\lowess_less_first\lowess' + str(sheep_num) + '_' + Y + '.csv')
                lowess = lowess.loc[lowess['daysFS'] < 20]
                plt.plot(lowess[X], lowess[Y], color='blue', linewidth=2, label=f'Sheep {sheep_num}')
            sheep_num_list = data.loc[(data['Entre_Date'] == '01/01/2023') & (data['type'] == 'diarrhea'), 'Tag'].astype(int)
            for i, sheep_num in enumerate(sheep_num_list):
                lowess = pd.read_csv(r'C:\Users\e3bom\OneDrive - post.bgu.ac.il\פרויקט גמר ומחקר\חלק יישומי\DATA ומודלים\DATA\Analysis_per _sheep\lowess_per_sheep\normalize\lowess_less_first\lowess' + str(sheep_num) + '_' + Y + '.csv')
                lowess = lowess.loc[lowess['daysFS'] < 20]
                plt.plot(lowess[X], lowess[Y], color='green', linewidth=2, label=f'Sheep {sheep_num}')
            sheep_num_list = data.loc[(data['Entre_Date'] == '01/01/2023') & (data['type'] == 'death'), 'Tag'].astype(int)
            for i, sheep_num in enumerate(sheep_num_list):
                lowess = pd.read_csv(r'C:\Users\e3bom\OneDrive - post.bgu.ac.il\פרויקט גמר ומחקר\חלק יישומי\DATA ומודלים\DATA\Analysis_per _sheep\lowess_per_sheep\normalize\lowess_less_first\lowess' + str(sheep_num) + '_' + Y + '.csv')
                first_value_y = lowess[Y][0]
                normalized_y = lowess[Y] - first_value_y
                first_value_x = lowess[X][0]
                normalized_x = lowess[X] - first_value_x
                x = normalized_x
                y = normalized_y
                plt.plot(x, y, color='red', linewidth=2, label=f'Sheep {sheep_num}')
            # Set labels and title for the plot
            plt.xlabel(X)
            plt.ylabel(Y)
            plt.title(' Comparison_less_first_with_exceptionuntil_20')
            # plt.legend()
            plt.savefig(r"C:\Users\e3bom\OneDrive - post.bgu.ac.il\פרויקט גמר ומחקר\חלק יישומי\DATA ומודלים\DATA\Analysis_per _sheep\plot_per sheep\_multi_lowess\Comparison_less_first_until_20" + Y + ".png")
            plt.show()
            return


def plot_simple():
    dataNum = pd.read_csv(
        r'C:\Users\e3bom\OneDrive - post.bgu.ac.il\פרויקט גמר ומחקר\חלק יישומי\DATA ומודלים\DATA\parameter examination\lowess\data_gap_lowess_madian_cal_wight.csv')
    frac_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    for frac in frac_list:
        column_name = f"{frac}_diff"

        # if X=='date':
        #     plt.xticks(rotation=90)
        #     plt.xticks(fontsize=6)
        plt.title("gap" + str(frac))
        x = dataNum['Unnamed: 0']
        y = dataNum[column_name]
        plt.xlabel('SheepNum')
        plt.ylabel(column_name)
        plt.scatter(x, y)
        plt.savefig(r"C:\Users\e3bom\OneDrive - post.bgu.ac.il\פרויקט גמר ומחקר\חלק יישומי\DATA ומודלים\DATA\parameter examination\lowess\gap_plots\\lowess_median" + column_name + ".png")
        plt.show()


def statistic_tests_W4_legs(W4_leg):
    W4_leg = W4_leg.dropna(subset=['Weight_4_legs', 'Weight_device'], how='any')

    #
    # # Check for outliers and remove them using threshold
    outlier_cols = W4_leg['Weight_device']
    for col in outlier_cols:
        W4_leg = W4_leg[W4_leg['Weight_device'] != '?']

    W4_leg.to_csv(r'C:\Users\e3bom\OneDrive - post.bgu.ac.il\פרויקט גמר ומחקר\חלק יישומי\DATA ומודלים\DATA\information_about_data\4_legs_W\C_W4_leg_22-02-23.csv')
    Table_Anova = pd.read_csv(
        r'C:\Users\e3bom\OneDrive - post.bgu.ac.il\פרויקט גמר ומחקר\חלק יישומי\DATA ומודלים\DATA\information_about_data\4_legs_W\Table_Anova.csv')
    model = ols('sheepNum ~ type', data=Table_Anova).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    # print the ANOVA table
    print("anova_table")
    print(anova_table)
    # specify the repeated measures design in the formula
    print("repeated measures ANOVA")
    model = ols('sheepNum ~ time + subject + subject:time', data=Table_Anova).fit()

    # generate the ANOVA table
    anova_table = sm.stats.anova_lm(model, typ=2)

    # print the ANOVA table
    print(anova_table)

    # # Perform the chi-square test
    # stat, p, dof, expected = chi2_contingency(contingency_table)
    #
    # # Print the results
    # print('Chi-square statistic:', stat)
    # print('P-value:', p)
    # print('Degrees of freedom:', dof)
    # print('Expected frequencies:', expected)

    # # Perform the t-test
    # stat, p = ttest_ind(W4_leg['Weight device'], W4_leg['Weight 4 legs'])
    #
    # # Print the results
    # print('T-statistic:', stat)
    # print('P-value:', p)

    # Perform the one-way ANOVA test


def full_plot(Num,DataBase):
    plot(Num, 'Scatter_lowess', 'daysFS', 'weight',DataBase)
    plot(Num, 'Scatter_lowess', 'daysFS', 'ml',DataBase)
    plot(Num, 'Scatter', 'daysFS', 'duration',DataBase)
    plot(Num, 'hist', 'daysFS', 'date-time',DataBase)
    plot(Num, 'Scatter', 'daysFS', 'cross',DataBase)

# --------------------------------------------------------------plot-------------------------------------------------------
downlead_file('ivri2')
# downlead_file('exception')
# Data_herb
data=pd.read_csv(r'C:\Users\e3bom\OneDrive - post.bgu.ac.il\פרויקט גמר ומחקר\חלק יישומי\DATA ומודלים\DATA\information_about_data\4_legs_W\C_W4_leg_22-02-23.csv')
sheep_num_list = data.loc[data['Entre_Date'] == '01/01/2023', 'Tag'].astype(int)

# sheep_num_list=['22723','22717']
# Iterate over the sheep numbers
# for i, sheep_num in enumerate(sheep_health_list):
for sheep_num in sheep_num_list:
    plot(sheep_num,'Scatter_lowess', 'daysFS', 'weight','ivri')
    # plot(sheep_num,'lowess_division_first', 'daysFS', 'weight','ivri')
    # plot(sheep_num,'lowess_less_first', 'daysFS', 'weight','ivri')

# plot('multi','multi_exception_less_first_with_exception', 'daysFS', 'weight')



# data = downlead_file('exception')
# parameter_exam('22/02/2023')
# gap_cal('median')
# plot_simple()
# plot('22709','daysFS','weight',lowess=True)
# full_plot('22674','exception')
# plot('multi', 'multi_exception', 'daysFS', 'weight')
# plot('250,017,033,503,954','multi_exception', 'daysFS', 'weight')
# plot('250,017,033,503,954','Scatter','daysFS','weight','TechCare')
# plot('509','multi_exception', 'daysFS', 'weight')
# plot('509','lowess', 'daysFS', 'weight' ,'ivri')
# plot('509','Scatter_lowess', 'daysFS', 'weight' ,'ivri')

# data=pd.read_csv(r'C:\Users\e3bom\OneDrive - post.bgu.ac.il\פרויקט גמר ומחקר\חלק יישומי\DATA ומודלים\DATA\parameter examination\lowess\data_gap_cal_wight.csv')
# plt.xlabel('Weight_4_legs')
# plt.title("'SheepNum'~abs(0.3_diff)/Weight_4_legs")
# plt.ylabel('abs(0.3_diff)')
# plt.scatter(data['Weight_4_legs'],data['abs(0.3_diff)/Weight_4_legs'])
# plt.savefig(r"C:\Users\e3bom\OneDrive - post.bgu.ac.il\פרויקט גמר ומחקר\חלק יישומי\DATA ומודלים\DATA\parameter examination\lowess\gap_plots\weight~diff\SheepNum'~abs(0.3_diff)%Weight_4_legs.png")
# plt.show()
# plt.title("abs(0.3_diff)~'SheepNum'")
# plt.ylabel('SheepNum')
# plt.xlabel('abs(0.3_diff)')
# plt.scatter(data['abs(0.3_diff)'],data['SheepNum'])
# plt.savefig(r"C:\Users\e3bom\OneDrive - post.bgu.ac.il\פרויקט גמר ומחקר\חלק יישומי\DATA ומודלים\DATA\parameter examination\lowess\gap_plots\sheepNum~diff\abs(0.3_diff)~Weight_4_legs.png")
# plt.show()
# plt.title("0.3_diff~'SheepNum'")
# plt.ylabel('SheepNum')
# plt.xlabel('0.3_diff')
# plt.scatter(data['0.3_diff'],data['SheepNum'])
# plt.savefig(r"C:\Users\e3bom\OneDrive - post.bgu.ac.il\פרויקט גמר ומחקר\חלק יישומי\DATA ומודלים\DATA\parameter examination\lowess\gap_plots\sheepNum~diff\0.3_diff~Weight_4_legs.png")
# plt.show()
# data_sorted = data.sort_values(by='0.6_diff')
# plt.title("'SheepNum'~0.6_diff")
# plt.xlabel('SheepNum')
# plt.ylabel('0.6_diff')
# plt.scatter(data['SheepNum'],data['0.6_diff'])
# plt.savefig(r"C:\Users\e3bom\OneDrive - post.bgu.ac.il\פרויקט גמר ומחקר\חלק יישומי\DATA ומודלים\DATA\parameter examination\lowess\gap_plots\sheepNum~diff\Weight_4_legs~0.6_diff.png")
# plt.show()
# plt.xlabel('SheepNum')
# plt.title("'SheepNum'~abs(0.6_diff)")
# plt.ylabel('abs(0.6_diff)')
# plt.scatter(data['SheepNum'],data['abs(0.6_diff)'])
# plt.savefig(r"C:\Users\e3bom\OneDrive - post.bgu.ac.il\פרויקט גמר ומחקר\חלק יישומי\DATA ומודלים\DATA\parameter examination\lowess\gap_plots\sheepNum~diff\Weight_4_legs~abs(0.6_diff).png")
# plt.show()
# plt.title("abs(0.6_diff)~'SheepNum'")
# plt.ylabel('SheepNum')
# plt.xlabel('abs(0.6_diff)')
# plt.scatter(data['abs(0.6_diff)'],data['SheepNum'])
# plt.savefig(r"C:\Users\e3bom\OneDrive - post.bgu.ac.il\פרויקט גמר ומחקר\חלק יישומי\DATA ומודלים\DATA\parameter examination\lowess\gap_plots\sheepNum~diff\abs(0.6_diff)~Weight_4_legs.png")
# plt.show()
# plt.title("0.6_diff~'SheepNum'")
# plt.ylabel('SheepNum')
# plt.xlabel('0.6_diff')
# plt.scatter(data['0.6_diff'],data['SheepNum'])
# plt.savefig(r"C:\Users\e3bom\OneDrive - post.bgu.ac.il\פרויקט גמר ומחקר\חלק יישומי\DATA ומודלים\DATA\parameter examination\lowess\gap_plots\sheepNum~diff\0.6_diff~Weight_4_legs.png")
# plt.show()