---
title: "Checking Survey Data"
date: 2019-07-05
header:
  image: /assets/images/checking/background.jpg
  teaser: /assets/images/checking/background.jpg
  caption: image from web
tags:
- survey
- pandas
- numpy
- missing data
categories:
- survey
---


# Checking survey data: examine rows, columns, data types, and missing values

In this post, I will cover the steps that I usually take when taking a first look at some survey data with Python. <br> 
This post uses data from 2016 OSMI Mental Health in Tech Survey, which is a publicly available [Kaggle dataset.](https://www.kaggle.com/osmi/mental-health-in-tech-2016)

#### Load Python packages and import data from a csv file

NumPy, Pandas, and Matplotlib are neccessary packages to perform data manipulation and visualization tasks. 


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```


```python
df = pd.read_csv('../input/IT_mental_health.csv')
```

#### Examine data shape


```python
df.shape
```




    (1433, 63)



The shape of the dataset indicates that there were n=1433 respondents (rows) who participated in the survey and 63 columns of information (columns) collected. That sounds like a pretty long questionnaire that survey methodologist would say no to, since long questionnaires lead to more response burden and that respondents tend to satisfice (e.g. speed through without careful thoughts, choose same answers for subsequent questions). However, it may be okay if there were skipping and branching implemented. Let's take a look at what the data really looks like.


```python
df.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Are you self-employed?</th>
      <th>How many employees does your company or organization have?</th>
      <th>Is your employer primarily a tech company/organization?</th>
      <th>Is your primary role within your company related to tech/IT?</th>
      <th>Does your employer provide mental health benefits as part of healthcare coverage?</th>
      <th>Do you know the options for mental health care available under your employer-provided coverage?</th>
      <th>Has your employer ever formally discussed mental health (for example, as part of a wellness campaign or other official communication)?</th>
      <th>Does your employer offer resources to learn more about mental health concerns and options for seeking help?</th>
      <th>Is your anonymity protected if you choose to take advantage of mental health or substance abuse treatment resources provided by your employer?</th>
      <th>If a mental health issue prompted you to request a medical leave from work, asking for that leave would be:</th>
      <th>...</th>
      <th>If you have a mental health issue, do you feel that it interferes with your work when being treated effectively?</th>
      <th>If you have a mental health issue, do you feel that it interferes with your work when NOT being treated effectively?</th>
      <th>What is your age?</th>
      <th>What is your gender?</th>
      <th>What country do you live in?</th>
      <th>What US state or territory do you live in?</th>
      <th>What country do you work in?</th>
      <th>What US state or territory do you work in?</th>
      <th>Which of the following best describes your work position?</th>
      <th>Do you work remotely?</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>26-100</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>Not eligible for coverage / N/A</td>
      <td>NaN</td>
      <td>No</td>
      <td>No</td>
      <td>I don't know</td>
      <td>Very easy</td>
      <td>...</td>
      <td>Not applicable to me</td>
      <td>Not applicable to me</td>
      <td>39</td>
      <td>Male</td>
      <td>United Kingdom</td>
      <td>NaN</td>
      <td>United Kingdom</td>
      <td>NaN</td>
      <td>Back-end Developer</td>
      <td>Sometimes</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>6-25</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Somewhat easy</td>
      <td>...</td>
      <td>Rarely</td>
      <td>Sometimes</td>
      <td>29</td>
      <td>male</td>
      <td>United States of America</td>
      <td>Illinois</td>
      <td>United States of America</td>
      <td>Illinois</td>
      <td>Back-end Developer|Front-end Developer</td>
      <td>Never</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>6-25</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>No</td>
      <td>NaN</td>
      <td>No</td>
      <td>No</td>
      <td>I don't know</td>
      <td>Neither easy nor difficult</td>
      <td>...</td>
      <td>Not applicable to me</td>
      <td>Not applicable to me</td>
      <td>38</td>
      <td>Male</td>
      <td>United Kingdom</td>
      <td>NaN</td>
      <td>United Kingdom</td>
      <td>NaN</td>
      <td>Back-end Developer</td>
      <td>Always</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 63 columns</p>
</div>



#### Check rows

As a common practice, I usually first check rows and see if there are any duplicated records. Duplication in survey data suggests that respondents entered exactly same values across all survey questions. For this study, this is likely an error since there are quite a number of questions so chance of natural duplication should be low.


```python
# check for duplication
df.duplicated().sum()
```




    0



#### Check columns

The first impression after looking at the head of the dataset is that the columns have the full question texts as their names. We can first check whether that's the case, and if so, we could assign appropriate names to the columns.


```python
question_list = df.columns.to_list()
len(question_list)
```




    63




```python
question_list[0:5] # only show the first five as an example
```




    ['Are you self-employed?',
     'How many employees does your company or organization have?',
     'Is your employer primarily a tech company/organization?',
     'Is your primary role within your company related to tech/IT?',
     'Does your employer provide mental health benefits as part of healthcare coverage?']



By checking `df.columns` we know that indeed the data columns are named with their original question texts. We usually would want the column names to be concise yet meaningful. Here we could rename the columns by assigning short phrases as names.


```python
# assign variable names
df.columns = ["self", "employee", "type", "IT", "benefit", "options", "discussion", "learn", "anoymity", "leave", "mental_employer", "physical_employer",
              "mental_coworker", "mental_super", "as_serious", "consequence", "coverage", "resources", "reveal_client", "impact", "reveal_coworker",
              "impact_neg", "productivity", "time_affected", "ex", "coverage_ex", "options_ex", "discussion_ex", "learn_ex", "anoymity_ex", "mental_employer_ex",
              "physical_employer_ex", "mental_coworker_ex", "mental_super_ex", "as_serious_ex", "consequence_ex", "physical_interview", "physical_interview_y", 
              "mental_interview", "mental_interview_y", "hurt", "negative", "share", "unsupport", "experience", "history", "past", "mental_disorder",
              "conditions_diagosed1", "conditions_believe", "diagnose", "conditions_diagosed2", "treatment", "interfere", "interfere_nottreated",
              "age", "gender", "country_live", "state", "country_work", "state_work", "position", "remote"]
```

Looking at `df.head()`, another thing I noticed is that this dataset does not come with a unique identifier. It probably won't affect the analysis work too much, but may still cause confusion when for example sorting and/or merging is involved. We could simply use index of the rows as respondent ID.


```python
df["ID"] = df.index + 1
```

#### Missing data and data type

The first few rows of the dataframe also shows a number of `NaN`, suggesting existence of missing values. There are several missing scenarios for survey data: 1) missing due to legitimate skips such as programming skips and branching; 2) missing due to intentional skips/don't know/refused; 3) systematic missing due to programming errors, usually checked against questionnaire and programming spec. Ideally we would want to distinguish between these missingness to make better analysis decisions.

Checking variable types is also an essential step before carrying out any analysis but is often overlooked. Different types could imply different statistical and visualization choices (will be discussed in more detail in the next post). Here we take a look at the types of variables to help us better understand the questionnaire (in the case no questionnaire is given).


```python
# create a function for summarizing n, number of missing values, variable types, and values of each variable
def summarize_column(data):
    valuelist =[]
    for col in data.columns:
        valuelist.append(data[col].unique())
    summarydic = {'variable_name': data.columns, 
                  'n': data.notnull().sum().to_list(),
                  'n_miss': data.isnull().sum().to_list(),
                  'values': valuelist,
                  'variable_type': data.dtypes}
    summarydf = pd.DataFrame(summarydic)
    return summarydf
```


```python
summarytable = summarize_column(df)
summarytable[0:5] # only show the first five as an example
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>variable_name</th>
      <th>n</th>
      <th>n_miss</th>
      <th>values</th>
      <th>variable_type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>self</th>
      <td>self</td>
      <td>1433</td>
      <td>0</td>
      <td>[0, 1]</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>employee</th>
      <td>employee</td>
      <td>1146</td>
      <td>287</td>
      <td>[26-100, 6-25, nan, More than 1000, 100-500, 5...</td>
      <td>object</td>
    </tr>
    <tr>
      <th>type</th>
      <td>type</td>
      <td>1146</td>
      <td>287</td>
      <td>[1.0, nan, 0.0]</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>IT</th>
      <td>IT</td>
      <td>263</td>
      <td>1170</td>
      <td>[nan, 1.0, 0.0]</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>benefit</th>
      <td>benefit</td>
      <td>1146</td>
      <td>287</td>
      <td>[Not eligible for coverage / N/A, No, nan, Yes...</td>
      <td>object</td>
    </tr>
  </tbody>
</table>
</div>



Looking at the table above I'm already seeing mis-categorize of variable type. For example, the variable "type" and "IT" look like categorical variables that take on levels 1 and 0. We could further edit those back to the desired type.


```python
# create a plot showing the number of missing values summarized by variable
summarytable['n_miss'].value_counts()[1:].plot(kind='bar', figsize=(13,9))
plt.xlabel("number of missing data", labelpad=10, fontsize=14)
plt.xticks(rotation=360)
plt.ylabel("Number of variables", labelpad=10, fontsize=14)
plt.title("Number of variables that have certain number of missing values", y=1.02, fontsize=15)
plt.show()
```


![png](/assets/images/checking/output_27_0.png)


The above bar chart shows some missing pattern - we can see that there are a number of variables that have 287, 169, and 1146 missing data, which implies that this kind of missingness could be due to legitimate skip of survey questions. The rest of the missing values are only specific to certain variables, which could be due to respondent skipping that question (don't know or refusal). For now, we don't want to recode or impute for missing values. These procedures are usually carried out with specific analysis intentions.
