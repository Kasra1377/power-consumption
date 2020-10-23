import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import seaborn as sns
from PIL import Image

import plotly.express as px
import plotly.graph_objs as go
import chart_studio.plotly as py
import cufflinks as cf

from plotly.offline import download_plotlyjs , init_notebook_mode
init_notebook_mode(connected = True)
cf.go_offline()

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor , GradientBoostingRegressor
from sklearn import model_selection , preprocessing

import xgboost as xgb
from xgboost import plot_importance, plot_tree

import warnings
warnings.filterwarnings('ignore')
#-----------------------------------------------

st.write('# USA Power Consumption Prediction Using Machine Learning Algorithms')
st.write('In this WebApp I am going to predict USA power consumption using various Machine Learning algorithms such as XGBoost.')

st.write('---')

html_table = """<a id="top"></a>
<div class="list-group" id="list-tab" role="tablist">
<h3 class="list-group-item list-group-item-action active" data-toggle="list"  role="tab" aria-controls="home">Table Of Contents</h3>"""
st.markdown(html_table, unsafe_allow_html = True)

html_content = ("""

<font color="black" size=+1><b>Introduction</b></font>
* [1. PJM Interconnection](#0)

<font color="black" size=+1><b>Initial Overview of Dataset</b></font>
* [1. Importing Essential Libraries](#2)
* [2. Importing the Dataset](#3)
* [3. General Information of the Dataset](#4)
* [4. Dataset Manipulation and Engineering Features](#4)

<font color="black" size=+1><b>Analyzing the USA Power Consumption Dataset</b></font>
* [1. Power Consumption Per Date](#5)
* [2. Power Consumption by Week of Year](#6)
* [3. Power Consumption by Month](#7)
* [4. Power Consumption by Weekday](#8)
* [5. Power Consumption Per Hour](#7)
* [6. Power Consumption Per Season](#7)

<font color="black" size=+1><b>Predicting Power Consumption</b></font>
* [1. Predicting with XGBoost Regressor](#9)
* [2. Predicting with Linear Regression](#10)
* [3. Predicting with Decision Tree](#11)
* [4. Predicting with Random Forest](#12)
* [5. Predicting with Gradient Boosting Machine](#13)

""")
st.markdown(html_content, unsafe_allow_html = True)

st.subheader('PJM Interconnection')

html_button = '<a href="#top" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover" title="go to Colors">Go to Table of Contents</a>'
st.markdown(html_button, unsafe_allow_html = True)

image = Image.open('C:/Users/Kasra/Desktop/images/Second Project/13.png')
st.image(image , use_column_width = True)

st.write('PJM Interconnection LLC (PJM) is a regional transmission organization (RTO) in the United States. It is part of the Eastern Interconnection grid operating an electric transmission system serving all or parts of Delaware, Illinois, Indiana, Kentucky, Maryland, Michigan, New Jersey, North Carolina, Ohio, Pennsylvania, Tennessee, Virginia, West Virginia, and the District of Columbia.')

image = Image.open('C:/Users/Kasra/Desktop/images/Second Project/14.png')
st.image(image , use_column_width = True)

st.write('In this kernel we are going to analyze the dataset that PJM organization provided for us...')
st.write('[PJM Organization Official Website](www.pjm.com)')
st.write('---')
st.subheader('Initial Overview of Dataset')
st.subheader('Importing Essential Libraries')

with st.echo():
    import streamlit as st
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    plt.style.use('fivethirtyeight')
    import seaborn as sns

    import plotly.express as px
    import plotly.graph_objs as go
    import chart_studio.plotly as py
    import cufflinks as cf

    from plotly.offline import download_plotlyjs , init_notebook_mode
    init_notebook_mode(connected = True)
    cf.go_offline()

    from sklearn.linear_model import LinearRegression
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor , GradientBoostingRegressor
    from sklearn import model_selection , preprocessing

    import xgboost as xgb
    from xgboost import plot_importance, plot_tree

    import warnings
    warnings.filterwarnings('ignore')

st.write('---')

st.subheader('Importing the Dataset')

html_button = '<a href="#top" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover" title="go to Colors">Go to Table of Contents</a>'
st.markdown(html_button, unsafe_allow_html = True)

st.write('Lets import the dataset:')

with st.echo():
    PJMW_hourly = pd.read_csv('E:/Datasets/Power Consumption US/PJMW_hourly.csv')
    PJMW_hourly

st.write('* This dataset has `143206` instances and `2` features.It is worth mentioning that this power consumption recorded every **1 hour**.')

st.write("""The columns of this dataset are:
> **1. Datetime** : The particular data and time that this dataset was recorded


> **2. PJMW_MW**  : Which is the amount of power consumption in 1 hour period of that date and time
""")

st.write('---')
st.write('Lets take the first date that the power consumption was recorded:')

with st.echo():
    PJMW_hourly['Datetime'].min()

st.write('> ' , PJMW_hourly['Datetime'].min())

st.write('Lets take the last date that the power consumption was recorded:')

with st.echo():
    PJMW_hourly['Datetime'].min()

st.write('> ' , PJMW_hourly['Datetime'].max())
st.write('* As this two dates show, the data was recorded on `2002-04-01 01:00:00` for the **first time** and was recorded on `2018-08-03 00:00:00` for the **last time**.')

st.write('---')
st.subheader('General Information of The Dataset')

html_button = '<a href="#top" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover" title="go to Colors">Go to Table of Contents</a>'
st.markdown(html_button, unsafe_allow_html = True)

st.write('In this section we are going to get the general information of dataset such as columns , number of null values , ...')

st.write('Lets get the information of dataset columns:')

with st.echo():
    PJMW_hourly.info()

st.write(PJMW_hourly.info())
st.write('* As this table shows, this dataset has 2 columns which one of them(`Datetime` column) has `object` or `string` datatype and the other(`PJMW_MW` column) has `float64` datatype.\n')

st.write('---')
st.write('In the next step lets get the general information of `PJMW_MW` column:')

with st.echo():
    PJMW_hourly.describe()

st.write(PJMW_hourly.describe())
st.write('* As this data shows, `PJMW_MW` column has **143206** instances which we have just dscovered before.In addition to this,mean of all instances is about **5602** and the standard deviation of them is around **980**.Minimum and the maximum of the instances is **487** and **9594** in respect.')

st.write('---')
st.write('And get the number of null values of each column:')

with st.echo():
    no_of_rows = PJMW_hourly.shape[0]
    percentage_of_missing_data = PJMW_hourly.isnull().sum()/no_of_rows
    percentage_of_missing_data

st.write('* As we can see , neither of `Datetime` or `PJMQ_MW` columns does not have null values which is satisfactory.')

st.write('---')
st.subheader('Dataset Manipulation and Engineering Features')

html_button = '<a href="#top" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover" title="go to Colors">Go to Table of Contents</a>'
st.markdown(html_button, unsafe_allow_html = True)

st.write('As you can guess this dataset this capability that extract many features from it.We can extract many feaures from `Datetime` column such as `Year` , `Month` , `Weekday` and etc.')
st.write('\nNow again lets take a look at our dataset:')

PJMW_hourly = PJMW_hourly.sort_values(by = 'Datetime').reset_index()
PJMW_hourly.drop(columns = {'index'} , inplace = True)
PJMW_hourly

with st.echo():
    PJMW_hourly

st.write('First of all lets change `Datetime` column format into `Datetime` format:')


st.write('Next step,lets decompose `Datetime` column into  `DATE` , `TIME` , `HOUR` , `DAY OF MONTH` , `WEEKDAY` , `WEEKDAY_NAME` , `MONTH` , `MONTH NAME` , `WEEK OF YEAR` , `YEAR` and `SEASON`:')
with st.echo():
    pd.Series([pd.Timestamp("2002-09-21 23:00:00")]).dt.tz_localize('US/Central' , ambiguous = True)
    PJMW_hourly['Datetime'] = pd.to_datetime(PJMW_hourly['Datetime'])
    PJMW_hourly['DATE'] = PJMW_hourly['Datetime'].apply(lambda x : x.date())

    PJMW_hourly['TIME'] = PJMW_hourly['Datetime'].apply(lambda x : x.time())
    PJMW_hourly['HOUR'] = PJMW_hourly['Datetime'].apply(lambda x : x.hour)
    PJMW_hourly['DAY OF MONTH'] = PJMW_hourly['Datetime'].apply(lambda x : x.day)
    PJMW_hourly['WEEKDAY'] = PJMW_hourly['Datetime'].apply(lambda x : x.weekday())
    PJMW_hourly['WEEKDAY_NAME'] = PJMW_hourly['Datetime'].apply(lambda x : x.day_name())
    PJMW_hourly['MONTH'] = PJMW_hourly['Datetime'].apply(lambda x : x.month)
    PJMW_hourly['MONTH NAME'] = PJMW_hourly['Datetime'].apply(lambda x : x.month_name())
    PJMW_hourly['WEEK OF YEAR'] = PJMW_hourly['Datetime'].apply(lambda x : x.weekofyear)
    PJMW_hourly['YEAR'] =  PJMW_hourly['Datetime'].apply(lambda x : x.year)
    PJMW_hourly['date_offset'] = (PJMW_hourly.Datetime.dt.month * 100 + PJMW_hourly.Datetime.dt.day - 320) % 1300

    PJMW_hourly['SEASON'] = pd.cut(PJMW_hourly['date_offset'] , [0, 300, 602, 900, 1300] ,
                                labels=['Spring', 'Summer', 'Fall', 'Winter'])

    PJMW_hourly.drop(columns = {'date_offset'} , inplace = True)

PJMW_hourly = pd.read_csv('E:/Datasets/Power Consumption US/usa.csv')
PJMW_hourly.drop(columns = {'Unnamed: 0'} , inplace = True)

st.write('And we will get the following result:')
st.write(PJMW_hourly.head())

st.write('---')
st.subheader('Analyzing The USA Power Consumption Dataset')

html_button = '<a href="#top" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover" title="go to Colors">Go to Table of Contents</a>'
st.markdown(html_button, unsafe_allow_html = True)

st.write('In this section we are going to answer the most frequent questions that we ususaly encounter with.')

st.subheader('1. Power Consumption Per Date')

html_button = '<a href="#top" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover" title="go to Colors">Go to Table of Contents</a>'
st.markdown(html_button, unsafe_allow_html = True)

st.write('First we have to group the `PJMW_MW` by `DATE`:')

with st.echo():
    day_consumption = PJMW_hourly.groupby(['DATE'])['PJMW_MW'].mean().reset_index()
    day_consumption.columns = ['DATE' , 'PJWM_DEMAND_MW']
    day_consumption

st.write('And plot the resulting dataset:')

with st.echo():
    fig = px.line(
        day_consumption,
        x = 'DATE',
        y = 'PJWM_DEMAND_MW',
        title = 'Power Consumption Per Date',
        labels={
                "DATE": "Date",
                "PJWM_DEMAND_MW": "Power Consumption (MW)",
                }
    )
    fig.show()

st.plotly_chart(fig, use_container_width = True)

st.write('* As this line graph illustrates , we have the data points since **2004** untill **2018**.We have many fluctuations in this graph and this means that our power consumption depends on various factors such as `Season` ,  `Hour` , `Month` etc.For instance , our peak values in this graph takes place in **holidays** or in the other word takes place in **winter** and **summer**.This is because in summer many heating and cooling systems are ON and this systems consume a lot of power.On the other hand in **fall** and **late spring and late summer** we have **minimum** power consumption.')

st.write('---')
st.subheader('2. Power Consumption by Week of Year')

html_button = '<a href="#top" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover" title="go to Colors">Go to Table of Contents</a>'
st.markdown(html_button, unsafe_allow_html = True)

st.write('First we have to group the `PJMW_MW` by `WEEK OF YEAR`:')

with st.echo():
    week_num_consumption = PJMW_hourly.groupby(['WEEK OF YEAR'])['PJMW_MW'].mean().reset_index().sort_values(by = 'WEEK OF YEAR')
    week_num_consumption.columns = ['WEEK OF YEAR' , 'PJMW_DEMAND_MW']
    week_num_consumption

st.write('\n And plot the previous dataset:')
with st.echo():
    fig = px.line(
        week_num_consumption,
        x = 'WEEK OF YEAR',
        y = 'PJMW_DEMAND_MW',
        title = 'Power Consumption by each Week of Year',
        labels={
                "WEEK OF YEAR": "Week of Year",
                "PJMW_DEMAND_MW": "Power Consumption (MW)",
                }
    )
    fig.show()
st.plotly_chart(fig, use_container_width = True)

st.write('---')

st.subheader('3. Power Consumption by Month')

html_button = '<a href="#top" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover" title="go to Colors">Go to Table of Contents</a>'
st.markdown(html_button, unsafe_allow_html = True)

st.write('First we have to group the `PJMW_MW` by `MONTH` column:')

with st.echo():
    month_consumption = PJMW_hourly.groupby(['MONTH' , 'MONTH NAME'])['PJMW_MW'].mean().reset_index().sort_values(by = 'MONTH')
    month_consumption.columns = ['MONTH' , 'MONTH NAME' , 'PJMW_DEMAND_MW']
    month_consumption

st.write('\n And plot the previous dataset:')
with st.echo():
    fig = px.line(
        month_consumption,
        x = 'MONTH NAME',
        y = 'PJMW_DEMAND_MW',
        title = 'Power Consumption Per Month',
        labels={
                "MONTH NAME": "Month",
                "PJMW_DEMAND_MW": "Power Consumption (MW)",
                }

    )
    fig.show()

st.plotly_chart(fig, use_container_width = True)
st.write('* As this graph shows,according to what we have just discovered before , the peak value takes place in **Janurary**(Holiday time) and **May** and **April** have minimum values and again this graph significantly risen up untill **July** and **Augest**.Because this months are in summer and in summer the power consumption will be risen up because of coolant systems.And again the graph significantly drops untill **October** and risen up untill **December**.')

st.write('---')
st.write('Lets illustrate this graph in the other way:')

st.write('\n And plot the previous dataset:')
with st.echo():
    fig = px.bar(
        month_consumption,
        x = 'MONTH NAME',
        y = 'PJMW_DEMAND_MW',
        color = 'PJMW_DEMAND_MW',
        title = 'Power Consumption Per Month',
        labels={
                "MONTH NAME": "Month",
                "PJMW_DEMAND_MW": "Power Consumption (MW)",
                }
    )

    fig.add_scatter(
        x = month_consumption['MONTH NAME'],
        y = month_consumption['PJMW_DEMAND_MW'],
        name = '',

    )

    fig.show()

st.plotly_chart(fig, use_container_width = True)

st.subheader('4. Power Consumption by Week Days')

html_button = '<a href="#top" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover" title="go to Colors">Go to Table of Contents</a>'
st.markdown(html_button, unsafe_allow_html = True)

st.write('First we have to group the `PJMW_MW` by `WEEKDAY` column:')

with st.echo():
    weekday_consumption = PJMW_hourly.groupby(['WEEKDAY' , 'WEEKDAY_NAME'])['PJMW_MW'].mean().reset_index().sort_values(by = 'WEEKDAY')
    weekday_consumption.columns = ['WEEKDAY' , 'WEEKDAY_NAME' , 'PJMW_DEMAND_MW']
    weekday_consumption

st.write('\n And plot the previous dataset:')
with st.echo():
    fig = px.line(
        weekday_consumption,
        x = 'WEEKDAY_NAME',
        y = 'PJMW_DEMAND_MW',
        title = 'Power Consumption Per Week Days',
        labels={
                "WEEKDAY_NAME": "Week Day",
                "PJWM_DEMAND_MW": "Power Consumption (MW)",
                }
    )

    fig.show()

st.plotly_chart(fig, use_container_width = True)
st.write('* As this graph shows, on weekdays(`Monday` , `Tuesday` , `Wednesday` , `Thursday` and `Friday`) the chart has its peak value, because in these days many factories and offices are open and in this palce a lot of power are consumed.On the other hand we have minimum power consumption on weekend(`Saturday` and `Sunday`).')

st.write('---')
st.subheader('5. Power Consumption per Hours')

html_button = '<a href="#top" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover" title="go to Colors">Go to Table of Contents</a>'
st.markdown(html_button, unsafe_allow_html = True)

st.write('First we have to group the `PJMW_MW` by `HOUR` column:')

with st.echo():
    hour_consumption = PJMW_hourly.groupby(['HOUR'])['PJMW_MW'].mean().reset_index()
    hour_consumption.columns = ['HOUR' , 'PJMW_DEMAND_MW']
    hour_consumption

st.write('\n And plot the previous dataset:')

with st.echo():
    fig = px.line(
        hour_consumption,
        x = 'HOUR',
        y = 'PJMW_DEMAND_MW',
        title = 'Power Consumption Per Hour',
        labels={
                "HOUR": "Hour",
                "PJMW_DEMAND_MW": "Power Consumption (MW)",
                }
    )

    fig.show()

st.plotly_chart(fig , use_container_width = True)
st.write('* As this graph represents, we have minimum consumption value at **midnight** and when we are getting closer to `noon` and `afternoon` we will get more power consumption which its peak value occurs at **7 PM**.')

st.write('---')
st.subheader('6. Power Consumption per Seasons')

html_button = '<a href="#top" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover" title="go to Colors">Go to Table of Contents</a>'
st.markdown(html_button, unsafe_allow_html = True)

st.write('First we have to group the `PJMW_MW` by `SEASON` column:')

with st.echo():
    season_consumption = PJMW_hourly.groupby(['SEASON'])['PJMW_MW'].mean().reset_index().sort_values(by = 'PJMW_MW')
    season_consumption.columns = ['SEASON' , 'PJMW_DEMAND_MW']
    season_consumption

st.write('\n And plot the previous dataset:')

with st.echo():
    fig = px.pie(
        season_consumption,
        values = 'PJMW_DEMAND_MW',
        names = 'SEASON',
        title = 'Power Consumption Per Season',
    )
    fig.update_traces(textposition = 'inside', textinfo = 'percent + label')
    fig.show()

st.plotly_chart(fig , use_container_width = True)
st.write('* This graph shows that most of the power consumption occurs in `winter`.As we said before most of the holidays events takes place in this season and in addition to this many heating systems are working in this season.After `winter`, `summer` has the most consumptioon because of cooling systems.')

st.write('---')
st.write('* We can illustrate this concept with another distribution graph:')

with st.echo():
    season_month_consumption = PJMW_hourly.groupby(['SEASON' , 'MONTH NAME'])['PJMW_MW'].mean().reset_index().sort_values(by = 'PJMW_MW')
    season_month_consumption.columns = ['SEASON' , 'MONTH NAME' , 'PJMW_DEMAND_MW']
    season_month_consumption

with st.echo():
    fig = px.sunburst(
        season_month_consumption,
        path = ['SEASON','MONTH NAME'],
        values = 'PJMW_DEMAND_MW',
        color = 'PJMW_DEMAND_MW',
        title = 'Power Consumption Distribution by Season and Month of each Season',
    )

    fig.show()

st.plotly_chart(fig , use_container_width = True)

st.write('---')
st.subheader('Predicting Power Consumption')

html_button = '<a href="#top" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover" title="go to Colors">Go to Table of Contents</a>'
st.markdown(html_button, unsafe_allow_html = True)

st.write('Lets import our dataset once more and sorting indices:')

with st.echo():
    PJMW_hourly = pd.read_csv('E:/Datasets/Power Consumption US/PJMW_hourly.csv' , index_col = [0], parse_dates = [0])
    PJMW_hourly = PJMW_hourly.sort_index()

st.write('\nAnd droping duplicate indices and keeping the first rows:')
with st.echo():
    PJMW_hourly = PJMW_hourly[~PJMW_hourly.index.duplicated(keep = 'first')]

st.write('\nSpiliting the dataset into the training set and testing set by a `2017-01-01` date.In other words we have the data before `2017-01-01` and we are going to predict the power consumption after this date.')


PJMW_hourly_train = pd.read_csv('E:/Datasets/Power Consumption US/usa-train.csv')
PJMW_hourly_test = pd.read_csv('E:/Datasets/Power Consumption US/usa-test.csv' )

st.write('Training set:')
PJMW_hourly_train

st.write('Testing set:')
PJMW_hourly_test

st.write('---')
st.write('Lets merge the `PJMW_hourly_test` and `PJMW_hourly_train` datasets together:')

with st.echo():
    PJMW_hourly_train.set_index('Datetime' , inplace = True)
    PJMW_hourly_test.set_index('Datetime' , inplace = True)
    PJMW_hourly_test.rename(columns={'PJMW_MW': 'Testing Set'}).join(PJMW_hourly_train.rename(columns={'PJMW_MW': 'Training Set'}), how='outer')

st.write('And lets plot the merged datasets that included with `Training set` and `Testing Set`:')

with st.echo():
    PJMW_hourly_test.plot(figsize=(15,5), title='Power consumption', style='.')

image = Image.open('C:/Users/Kasra/Desktop/images/Second Project/1.png')
st.image(image , use_column_width = True)
#st.write('![alt text](https://img.techpowerup.org/201021/1.png "Power Consumption")')

st.write('* As we can see, we divided the graph into two major parts.One of them is training set which are shown by `red` color and another is testing set which can be seen by `blue` color.We are goinig to train our model by `Training Set` and make prediction by `Testing Set` and then evaluate our model accuracy.')

st.write('---')
st.write('Then we defined a function that divide the dataset into testing and training set for our X(features) and y(label):')
with st.echo():
    def create_features(df, label=None):

        df['date'] = df.index
        df['date'] = pd.to_datetime(df['date'])
        df['hour'] = df['date'].dt.hour
        df['dayofweek'] = df['date'].dt.dayofweek
        df['quarter'] = df['date'].dt.quarter
        df['month'] = df['date'].dt.month
        df['year'] = df['date'].dt.year
        df['dayofyear'] = df['date'].dt.dayofyear
        df['dayofmonth'] = df['date'].dt.day
        df['weekofyear'] = df['date'].dt.weekofyear

        X = df[['hour','dayofweek','quarter','month','year',
               'dayofyear','dayofmonth','weekofyear']]
        if label:
            y = df[label]
            return X , y
        return X

st.write('And then lets use this function:')

with st.echo():
    X_train , y_train = create_features(PJMW_hourly_train , label = 'PJMW_MW')
    X_test , y_test = create_features(PJMW_hourly_test , label = 'PJMW_MW')

st.write('---')
st.subheader('1. Predicting with XGBoost Regressor')

html_button = '<a href="#top" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover" title="go to Colors">Go to Table of Contents</a>'
st.markdown(html_button, unsafe_allow_html = True)

with st.echo():
    reg = xgb.XGBRegressor(n_estimators = 1000)
    reg.fit(X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            early_stopping_rounds = 50,
            verbose = False)
st.write('Lets print the model accuracy:')

with st.echo():
    accuracy = reg.score(X_test, y_test)
    print('The model accuracy is :' , round(accuracy , 4) * 100 , ' %')

st.write('> The model accuracy is :' , round(accuracy , 4) * 100 , ' %')
st.write('And plot the feature importance :')

with st.echo():
    plot_importance(reg , height=0.9)

image = Image.open('C:/Users/Kasra/Desktop/images/Second Project/2.png')
st.image(image , use_column_width = True)

st.write('* In this feature importance plot, If the feature has a high value(`day of year` and `year` for instance), it means that it has a big role in our model and the model mostly make the decisions and predictions by this feature and on the other side, if a feature has a low value(`quarter` and `month`) it means that its value does not affect on our model prediction at all and it is less important for our model.')
st.write('---')
st.write('Lets make predictions by `XGBoost` , concatnate the prediction with training set and plotting them:')

with st.echo():
    PJMW_hourly_test['MW_Prediction'] = reg.predict(X_test)
    PJMW_hourly_all = pd.concat([PJMW_hourly_test, PJMW_hourly_train], sort = False)
    PJMW_hourly_all[['PJMW_MW','MW_Prediction']].plot(figsize=(15, 5) , title = 'Power Consumption Prediction by XGBoost')
    plt.show()

image = Image.open('C:/Users/Kasra/Desktop/images/Second Project/3.png')
st.image(image , use_column_width = True)

st.write('* As we can see the actual data represented by **blue** color and predicted data represented by **red** color.As we can see this `XGBoost` algorithm has not a bad performance at all and we can count on this models predictions.')
st.write('---')
st.write('In the next step lets zoom in a particular month and see how this model performed:')

with st.echo():
    f, ax = plt.subplots(1)
    f.set_figheight(5)
    f.set_figwidth(15)
    PJMW_hourly_all[['MW_Prediction','PJMW_MW']].plot(ax = ax , style = ['-','.'] , color = ['red' , 'blue'])
    ax.set_xbound(lower = '2017-01-01', upper = '2017-02-01')
    ax.set_ylim(0 , 10000)
    plt.suptitle('January 2017 Forecast vs Actual')

image = Image.open('C:/Users/Kasra/Desktop/images/Second Project/4.png')
st.image(image , use_column_width = True)

st.write('---')

st.subheader('2. Predicting with Linear Regression')

html_button = '<a href="#top" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover" title="go to Colors">Go to Table of Contents</a>'
st.markdown(html_button, unsafe_allow_html = True)

with st.echo():
    linear_reg = LinearRegression()
    linear_reg.fit(X_train , y_train)
    linear_reg_acc = linear_reg.score(X_test , y_test)
    linear_reg_forecast = linear_reg.predict(X_test)

st.write('Lets print the model accuracy:')

with st.echo():
    accuracy = reg.score(X_test, y_test)
    print('The model accuracy is :' , round(linear_reg_acc , 4) * 100 , ' %')

st.write('> The model accuracy is :' , round(linear_reg_acc , 4) * 100 , ' %')

st.write('---')
st.write('Lets make predictions by `Linear Regression` , concatnate the prediction with training set and plotting them:')

with st.echo():
    PJMW_hourly_test['MW_Prediction'] = linear_reg_forecast
    PJMW_hourly_all = pd.concat([PJMW_hourly_test, PJMW_hourly_train], sort = False)
    PJMW_hourly_all[['PJMW_MW','MW_Prediction']].plot(figsize=(15, 5) , title =  'Power Consumption Prediction by Linear Regression')
    plt.show()

image = Image.open('C:/Users/Kasra/Desktop/images/Second Project/5.png')
st.image(image , use_column_width = True)

with st.echo():
    f, ax = plt.subplots(1)
    f.set_figheight(5)
    f.set_figwidth(15)
    PJMW_hourly_all[['MW_Prediction','PJMW_MW']].plot(ax = ax , style = ['-','.'], color = ['red' , 'blue'])
    ax.set_xbound(lower = '2017-01-01', upper = '2017-03-01')
    ax.set_ylim(0 , 10000)
    plt.suptitle('January and Feburary 2017 Forecast vs Actuals')

image = Image.open('C:/Users/Kasra/Desktop/images/Second Project/6.png')
st.image(image , use_column_width = True)
st.write('* As we can see this linear regression has a very low performance because of its linear algorithm.')

st.write('---')
st.subheader('3. Predicting with Decision Tree')

html_button = '<a href="#top" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover" title="go to Colors">Go to Table of Contents</a>'
st.markdown(html_button, unsafe_allow_html = True)

with st.echo():
    decision_tree = DecisionTreeRegressor(max_depth = 6)
    decision_tree.fit(X_train , y_train)
    decision_tree_forecast = decision_tree.predict(X_test)
    tree_acc = decision_tree.score(X_test , y_test)

st.write('Lets print the model accuracy:')

with st.echo():
    accuracy = reg.score(X_test, y_test)
    print('The model accuracy is :' , round(tree_acc , 4) * 100 , ' %')

st.write('> The model accuracy is :' , round(tree_acc , 4) * 100 , ' %')

st.write('---')

with st.echo():
    PJMW_hourly_test['MW_Prediction'] = decision_tree_forecast
    PJMW_hourly_all = pd.concat([PJMW_hourly_test, PJMW_hourly_train], sort = False)
    PJMW_hourly_all[['PJMW_MW','MW_Prediction']].plot(figsize=(15, 5) , title = 'Power consumption Prediction by Decision Tree')
    plt.show()

image = Image.open('C:/Users/Kasra/Desktop/images/Second Project/7.png')
st.image(image , use_column_width = True)


with st.echo():
    f, ax = plt.subplots(1)
    f.set_figheight(5)
    f.set_figwidth(15)
    PJMW_hourly_all[['MW_Prediction','PJMW_MW']].plot(ax = ax , style = ['-','.'], color = ['red' , 'blue'])
    ax.set_xbound(lower = '2017-01-01', upper = '2017-02-01')
    ax.set_ylim(0 , 10000)
    plt.suptitle('January 2017 Forecast vs Actuals')

image = Image.open('C:/Users/Kasra/Desktop/images/Second Project/8.png')
st.image(image , use_column_width = True)

st.write('---')
st.subheader('4. Predicting with Random Forest')

html_button = '<a href="#top" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover" title="go to Colors">Go to Table of Contents</a>'
st.markdown(html_button, unsafe_allow_html = True)

with st.echo():
    random_forest = RandomForestRegressor()
    random_forest.fit(X_train , y_train)
    random_forest_forecast = random_forest.predict(X_test)
    forest_acc = random_forest.score(X_test , y_test)

st.write('Lets print the model accuracy:')

with st.echo():
    accuracy = reg.score(X_test, y_test)
    print('The model accuracy is :' , round(forest_acc , 4) * 100 , ' %')

st.write('> The model accuracy is :' , round(forest_acc , 4) * 100 , ' %')

st.write('---')

with st.echo():
    PJMW_hourly_test['MW_Prediction'] = decision_tree_forecast
    PJMW_hourly_all = pd.concat([PJMW_hourly_test, PJMW_hourly_train], sort = False)
    PJMW_hourly_all[['PJMW_MW','MW_Prediction']].plot(figsize=(15, 5) , title = 'Power consumption Prediction by Decision Tree')
    plt.show()

image = Image.open('C:/Users/Kasra/Desktop/images/Second Project/9.png')
st.image(image , use_column_width = True)

with st.echo():
    f, ax = plt.subplots(1)
    f.set_figheight(5)
    f.set_figwidth(15)
    PJMW_hourly_all[['MW_Prediction','PJMW_MW']].plot(ax = ax , style = ['-','.'], color = ['red' , 'blue'])
    ax.set_xbound(lower = '2017-01-01', upper = '2017-02-01')
    ax.set_ylim(0 , 10000)
    plt.suptitle('January 2017 Forecast vs Actuals')

image = Image.open('C:/Users/Kasra/Desktop/images/Second Project/10.png')
st.image(image , use_column_width = True)

st.write('---')
st.subheader('5. Predicting with Gradient Boosting Machine')

html_button = '<a href="#top" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover" title="go to Colors">Go to Table of Contents</a>'
st.markdown(html_button, unsafe_allow_html = True)

with st.echo():
    gradient_boost = GradientBoostingRegressor()
    gradient_boost.fit(X_train , y_train)
    gradient_boost_forecast = gradient_boost.predict(X_test)
    gradient_acc = gradient_boost.score(X_test , y_test)

st.write('Lets print the model accuracy:')

with st.echo():
    accuracy = reg.score(X_test, y_test)
    print('The model accuracy is :' , round(gradient_acc , 4) * 100 , ' %')

st.write('> The model accuracy is :' , round(gradient_acc , 4) * 100 , ' %')

st.write('---')

with st.echo():
    PJMW_hourly_test['MW_Prediction'] = gradient_boost.predict(X_test)
    PJMW_hourly_all = pd.concat([PJMW_hourly_test, PJMW_hourly_train], sort = False)
    PJMW_hourly_all[['PJMW_MW','MW_Prediction']].plot(figsize=(15, 5), title = 'Power Consumption Prediction by Gradient Boosting Machine')
    plt.show()

image = Image.open('C:/Users/Kasra/Desktop/images/Second Project/11.png')
st.image(image , use_column_width = True)


with st.echo():
    f, ax = plt.subplots(1)
    f.set_figheight(5)
    f.set_figwidth(15)
    PJMW_hourly_all[['MW_Prediction','PJMW_MW']].plot(ax = ax , style = ['-','.'], color = ['red' , 'blue'])
    ax.set_xbound(lower = '2017-01-01', upper = '2017-02-01')
    ax.set_ylim(0 , 10000)
    plt.suptitle('January 2017 Forecast vs Actuals')

image = Image.open('C:/Users/Kasra/Desktop/images/Second Project/12.png')
st.image(image , use_column_width = True)
