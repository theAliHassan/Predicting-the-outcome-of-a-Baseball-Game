import pandas
import pandas as pd
import numpy as np
import seaborn as sns
from seaborn.utils import saturate
import statsmodels.api as sm
import statsmodels.stats.api as sms
from statsmodels.formula.api import ols
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm
import math
import missingno as msno

# %matplotlib inline

df = pd.read_csv('BB_Team.csv')
df.head()
df.columns
df.isnull().any().sum()

df = df.rename(index=str,
               columns={"Tm_x": "Team", "W-L%": "W/L", '#Bat': 'Num_Hitters', '2B': 'Doubles', '3B': 'Triples',
                        'OPS+': 'OPS_Plus'})
# create column for hits per game
df['H/G'] = df['H'] / df['G']

# create column for extra base hits
df['Extra_Base_Hits'] = df['Doubles'] + df['Triples'] + df['HR']

# create column for BABIP
df['BABIP'] = (df['H'] - df['HR']) / (df['AB'] - df['SO'] - df['HR'] + df['SF'])

# create column for interaction of OBP*SLG
df['OBP_times_SLG'] = df.OBP * df.SLG

df['Age_of_Hitters'] = pd.cut(df.BatAge, 4, labels=["Rookies", "Young_Aged", "Avg_Aged", "Vet_Aged"])

# drop rows where year is 1994/1995 due to MLB strike
df = df[df.Year != 1994]
df = df[df.Year != 1995]

# create df that does not include 2018
df_not_2018 = df[df.Year != 2018]
df_not_2018.columns
df_not_2018.head()
# create model that only includes 2018
df_2018 = df[df.Year == 2018]

df[['R', 'OBP', 'SLG', 'LOB', 'SF', 'OBP_times_SLG']].describe()


def distributionofRuns():
    # distributions of runs
    sns.distplot(df['R'])
    plt.title('distributionOFRuns')
    plt.savefig('distributionOFRuns.png')
    plt.show()


def distancePlot():
    sns.distplot(df['OBP'])
    plt.title('distancePlot')
    plt.savefig('distancePlot.png')
    plt.show()


# distribution of SLG
def distributionofSLG():
    sns.distplot(df['SLG'])
    plt.title('distributionSLG')
    plt.savefig('distributionSLG.png')
    plt.show()


def distancePlot2():
    sns.distplot(df['OBP_times_SLG'])
    plt.title('distancePlot3')
    plt.savefig('distancePlot3.png')
    plt.show()


# scatterplot of SLG and R
def ScatterPlot_SLG_R():
    print('r:', df['SLG'].corr(df['R']))
    # sns.scatterplot(df['SLG'], df['R'])
    sns.regplot(x=df['SLG'], y=df['R'], data=df)
    # r: 0.8980068452393458
    plt.title('scatterPlot_SLG_R')
    plt.savefig('scatterPlot_SLG_R.png')
    plt.show()


# scatterplot of R and W
def ScatterPlot_R_W():
    print('r:', df['SLG'].corr(df['OBP']))
    sns.regplot(x=df['SLG'], y=df['OBP'], data=df)
    # r: 0.7402891588555475
    plt.title('scatterPlot_R_W')
    plt.savefig('scatterPlot_R_W.png')
    plt.show()


def residualPlot():
    print('r:', df['PA'].corr(df['R']))
    sns.regplot(x=df['PA'], y=df['R'], data=df)
    # r: 0.8164089648336568
    plt.title('residualPlot')
    plt.savefig('residualPlot.png')
    plt.show()


def BoxPlot():
    sns.boxplot(x=df['PA'])
    plt.title('BoxPlot')
    plt.savefig('BoxPlot.png')
    plt.show()


# 3D scatter plot of OBP_times_SLG, PA, and R
def scatterPlot3D_OBP_SLG_PA_R():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    fig.set_size_inches(8, 5)

    x = df['OBP_times_SLG']
    y = df['PA']
    z = df['R']

    ax.scatter(x, y, z, c='r', marker='.')

    ax.set_xlabel('OBP_times_SLG')
    ax.set_ylabel('PA')
    ax.set_zlabel('R')

    plt.title('scatterPlot3D_OBP_SLG_PA_R')
    plt.savefig('scatterPlot3D_OBP_SLG_PA_R')
    plt.show()


# 3D scatter plot of LOB, OBP, and R
def scatterPlot3D_LOB_OBP_R():
    # 3D scatter plot of LOB, OBP, and R
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    fig.set_size_inches(8, 5)

    x = df['LOB']
    y = df['OBP']
    z = df['R']

    ax.scatter(x, y, z, c='r', marker='.')

    ax.set_xlabel('LOB')
    ax.set_ylabel('OBP')
    ax.set_zlabel('R')

    plt.title('scatterPlot3D_LOB_OBP_R.png')
    plt.savefig('scatterPlot3D_LOB_OBP_R.png')
    plt.show()


# Building Our Regression Model

# model for R ~ PA + LOB + OBP + OBP_times_SLG using the df_not_2018 dataframe
model = ols(formula='R ~ PA + LOB + OBP + OBP_times_SLG', data=df_not_2018).fit()


# print(model.summary())

def Goldfel_Quandt_test():
    sns.residplot(df['PA'], df['R'], lowess=True)
    plt.title('residualPlot_RM')
    plt.savefig('residualPlot_RM.png')
    plt.show()

    f1 = 'R~PA+LOB+OBP+OBP_times_SLG'
    model_2 = ols(formula=f1, data=df).fit()
    resid = model_2.resid
    fig2 = sm.graphics.qqplot(resid, dist=stats.norm, line='45', fit=True)
    plt.savefig('OLS.png')
    plt.show()

    # run Goldfeld Quandt test
    # test for heteroskedasticity
    # null hypothesis assumes homoskedacity
    names = ['F Statistic', 'p-value']
    test = sms.het_goldfeldquandt(model_2.resid, model_2.model.exog)
    list(zip(names, test))


def predict_runs_scored(PA, OBP_times_SLG, OBP, LOB):
    return float((-1867.7977 + (0.3672 * PA) + (-0.6483 * LOB) + (2148.2308 * OBP) + (2633.9704 * OBP_times_SLG)))


# function that will predict the season runs for each row
# returns a list of predictions
def get_predicted_vals(dataframe):
    predictions = []
    for index, row in dataframe.iterrows():
        pa, obp, lob, obp_times_slg = row['PA'], row['OBP'], row['LOB'], row['OBP_times_SLG']
        predicted_runs_scored = predict_runs_scored(pa, obp_times_slg, obp, lob)
        # print(predicted_runs_scored)
        predictions.append(predicted_runs_scored)
    return predictions


# create variable for list of predicted values
values_predicted = get_predicted_vals(df_not_2018)

# set new column equal to the list of predictions

df_not_2018.insert(2, 'Predicted_Runs', values_predicted)
df_not_2018.head()


def RegressionModel():
    print('=========================================')
    print('=========================================')
    print()
    runs_true = df_not_2018['R']
    runs_pred = df_not_2018['Predicted_Runs']
    from sklearn.metrics import mean_squared_error
    rmse = math.sqrt(mean_squared_error(runs_true, runs_pred))
    runs_mean = df_not_2018['R'].mean()
    accuracy = rmse / runs_mean

    print('percentage of error:', round((accuracy * 100), 3), '%')
    print()
    print('=========================================')
    print('=========================================')

    # TEST MODEL ON STATS FROM 2018
    values_predicted_2018 = get_predicted_vals(df_2018)
    df_2018.insert(2,'Predicted_Runs',values_predicted_2018)
    df_2018[['Team', 'Year', 'R', 'Predicted_Runs']].head()
    runs_true_2018 = df_2018['R']
    runs_pred_2018 = df_2018['Predicted_Runs']
    mean_squared_error = mean_squared_error(runs_true_2018, runs_pred_2018)
    print('=========================================')
    print('=========================================')
    print()
    print('Mean squared error:', mean_squared_error)
    rmse = math.sqrt(mean_squared_error)
    print()
    print('=========================================')
    print('=========================================')

    runs_mean = df_2018['R'].mean()

    # find average percent of error
    poe = rmse / runs_mean
    print('=========================================')
    print('=========================================')
    print()
    print('RMSE:', rmse, 'runs')
    print('2018 mean runs scored:', runs_mean, 'runs')
    print('percentage of error:', round((poe * 100), 3), '%')
    print()
    print('=========================================')
    print('=========================================')

    print()
    print('=========================================')
    print('=========================================')
    # Create column for residuals
    df_not_2018.insert(3,'Residual', df_not_2018['Predicted_Runs'] - df_not_2018['R'])
    # plot residuals
    sns.residplot(df_not_2018['Residual'], df_not_2018['R'], lowess=True)
    plt.title('PredictionAnalysis_ResidualPlot')
    plt.savefig('PredictionAnalysis_ResidualPlot.png')
    plt.show()


# Model conclusion:Runs ~ PA + LOB + OBP + OBP*SLG
# The model shows a high r-squared (.960) and seems to predict a team's
# number of runs per season fairly well. As seen in the test case, the
# Chicago Cubs scored a total 777 runs throughout 2001 the season.
# My model predicts that a team with those same seasonal statistics
# would score 768 runs. For this test case,
# there is only a difference of 9 runs over the entire season.
# This result equates to roughly 0.05 run difference per game (9 runs / 162 games).


# HYPOTHESIS TEST

# The New York Yankees have won the most
# World Series Championships within the span of 1990-2018
# Let's look into how they match up to the rest of the MLB in terms of runs scored.

# H0: The mean runs for NYY = the mean runs for of all of MLB<br>
# Ha: The mean runs for NYY > the mean runs for of all of MLB


yankees_sample = df[df['Abv'] == 'NYY']['R'].mean()
mlb_pop_mean = df['R'].mean()
mlb_pop_std = df['R'].std()

print('=========================================')
print('=========================================')
print()
print('Yankees Mean:', yankees_sample)
print('MLB Mean:', mlb_pop_mean)
print('MLB Std:', mlb_pop_std)
print()
print('=========================================')
print('=========================================')


def Test(sample_, pop_mean, pop_std):
    # With a z score of .95 we fail to reject the null hypothesis

    print('=========================================')
    print('=========================================')
    print()
    sample = sample_
    mean = pop_mean
    std = pop_std
    z_score = (sample - mean) / std
    print('Z Score:', z_score)
    if z_score >= 1.96:
        print('Reject null hypothesis')
    elif z_score <= -1.96:
        print('Reject null hypothesis')
    else:
        print('Fail to reject null hypothesis')

    print()
    print('=========================================')
    print('=========================================')
    return z_score

def Normaldistribution():

    data='OBP'
    mu=np.mean(df[data]) 
    std= np.std(df[data])
    plt.subplot(211)
    plt.title('Normal Distribution Curve OBP')
    plt.plot(df[data], norm.pdf(df[data], mu, std))
    plt.xlabel('OBP')


    data='SLG'
    mu=np.mean(df[data]) 
    std= np.std(df[data])
    plt.subplot(212)
    plt.title('Normal Distribution Curve SLG')
    plt.plot(df[data], norm.pdf(df[data], mu, std))
    plt.xlabel('SLG')
    plt.savefig('Normal_Distribution_Curve_OBP_SLG.png')
    plt.show()

 




#Initial Tests

distributionofRuns()
distancePlot()
distributionofSLG()
distancePlot2()
ScatterPlot_SLG_R()
ScatterPlot_R_W()
residualPlot()
BoxPlot()
scatterPlot3D_OBP_SLG_PA_R()
scatterPlot3D_LOB_OBP_R()

# REGRESSION MODEL TESTS

Goldfel_Quandt_test()
RegressionModel()
Test(yankees_sample, mlb_pop_mean, mlb_pop_std)


# Normal Distribution

Normaldistribution()