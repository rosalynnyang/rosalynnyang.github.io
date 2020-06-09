---
title: "Working with survey data: more data cleaning"
date: 2019-07-05
header:
  image: /assets/images/post3/background.jpeg
  teaser: /assets/images/post3/background.jpeg
  caption: image from web
tags:
- survey
- pandas
- numpy
- data cleaning
- pycountry
- outlier
categories:
- survey
---

I originally wanted to move onto discussion on descriptive statistics but figured there are actually a lot more data cleaning work that need to be done and are worth sharing. This is usually the case in the data science work that I'm involved where about 80% of my time were about some sort of cleaning and getting the data ready in the desired format for anlayses. <br>
Therefore, in this post, I'll cover some techniques and handy functions to use when dealing with specific types of data cleaning. <br>
Check out my [first post](https://rosalynnyang.github.io/survey/checking-survey-data/) and [second post](https://rosalynnyang.github.io/survey/variable-type-missing/) in the same series.


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pycountry_convert as pc
```


```python
df = pd.read_parquet('../output/df')
```

Suppose I now have a more targeted analysis goal, which is to look at how employer policies and workplace environment affect worker's mental health state. In other words, I want to exclude those that are self-employeed and consider only variables related to employer and workplace. 

#### Subsetting data

**On the row:** There are many ways to subset data and I'm used to what's listed below - just put the row filter condition in the `.loc` square bracket and we are good to go.


```python
df_s = df.loc[df['self']==0]
```

**On the column:** In terms of selection of columns/variables, just use double square bracket `[['variable name']]`:


```python
df_s = df_s[['ID', 'diagnose', 'employee', 'type', 'benefit', 'discussion', 'learn', 
             'anoymity', 'leave', 'mental_employer', 'physical_employer', 'mental_coworker', 'mental_super', 
             'as_serious', 'consequence', 'age', 'gender', 'country_live', 'state', 'position', 'remote']]
```


```python
df_s.shape
```




    (1146, 21)



#### Check raw frequencies

After getting the data into the shape I want, I usually just check the raw frequencies of the variables and see if there's anything outstanding that needs my attention.

I like Python `list` and I think it is great for storing and organizing different objects. Here I take out the variables that are especially relevant to employer policies and workplace environment and save them into the list called `employer_cols` and demographics variables are saved in `demographic_cols` list.


```python
employer_cols = ['employee', 'type', 'benefit', 'discussion', 'learn', 'anoymity', 'leave', 
                 'mental_employer', 'physical_employer', 'mental_coworker', 'mental_super', 
                 'as_serious', 'consequence']

demographic_cols = ['age', 'gender', 'country_live', 'state', 'position', 'remote']
```

A quick `for loop` and `value_counts()` can give some raw frequencies output:


```python
for col in employer_cols:
    print(f"--------------frequency table of {col}: ---------------------")
    print(df_s[col].value_counts(dropna=False).sort_index())
```

    --------------frequency table of employee: ---------------------
    1-5                60
    100-500           248
    26-100            292
    500-1000           80
    6-25              210
    More than 1000    256
    Name: employee, dtype: int64
    --------------frequency table of type: ---------------------
    0.0    263
    1.0    883
    Name: type, dtype: int64
    --------------frequency table of benefit: ---------------------
    I don't know                       319
    No                                 213
    Not eligible for coverage / N/A     83
    Yes                                531
    Name: benefit, dtype: int64
    --------------frequency table of discussion: ---------------------
    I don't know    103
    No              813
    Yes             230
    Name: discussion, dtype: int64
    --------------frequency table of learn: ---------------------
    I don't know    320
    No              531
    Yes             295
    Name: learn, dtype: int64
    --------------frequency table of anoymity: ---------------------
    I don't know    742
    No               84
    Yes             320
    Name: anoymity, dtype: int64
    --------------frequency table of leave: ---------------------
    I don't know                  150
    Neither easy nor difficult    178
    Somewhat difficult            199
    Somewhat easy                 281
    Very difficult                118
    Very easy                     220
    Name: leave, dtype: int64
    --------------frequency table of mental_employer: ---------------------
    Maybe    487
    No       438
    Yes      221
    Name: mental_employer, dtype: int64
    --------------frequency table of physical_employer: ---------------------
    Maybe    268
    No       837
    Yes       41
    Name: physical_employer, dtype: int64
    --------------frequency table of mental_coworker: ---------------------
    Maybe    479
    No       392
    Yes      275
    Name: mental_coworker, dtype: int64
    --------------frequency table of mental_super: ---------------------
    Maybe    382
    No       336
    Yes      428
    Name: mental_super, dtype: int64
    --------------frequency table of as_serious: ---------------------
    I don't know    493
    No              303
    Yes             350
    Name: as_serious, dtype: int64
    --------------frequency table of consequence: ---------------------
    No     1048
    Yes      98
    Name: consequence, dtype: int64
    


```python
for col in demographic_cols:
    print(f"--------------frequency table of {col}: ---------------------") 
    print(df[col].value_counts(dropna=False).sort_index())
```

    --------------frequency table of age: ---------------------
    3       1
    15      1
    17      1
    19      4
    20      6
    21     15
    22     32
    23     24
    24     42
    25     44
    26     64
    27     63
    28     74
    29     79
    30     94
    31     82
    32     72
    33     69
    34     69
    35     74
    36     50
    37     59
    38     54
    39     55
    40     36
    41     24
    42     29
    43     30
    44     31
    45     27
    46     22
    47     14
    48      9
    49     13
    50      9
    51      7
    52      7
    53      3
    54      7
    55     12
    56      5
    57      4
    58      1
    59      2
    61      2
    62      1
    63      4
    65      1
    66      1
    70      1
    74      1
    99      1
    323     1
    Name: age, dtype: int64
    --------------frequency table of gender: ---------------------
     Female                                                                                                                                                            1
    AFAB                                                                                                                                                               1
    Agender                                                                                                                                                            2
    Androgynous                                                                                                                                                        1
    Bigender                                                                                                                                                           1
    Cis Male                                                                                                                                                           1
    Cis female                                                                                                                                                         1
    Cis male                                                                                                                                                           1
    Cis-woman                                                                                                                                                          1
    Cisgender Female                                                                                                                                                   1
    Dude                                                                                                                                                               1
    Enby                                                                                                                                                               1
    F                                                                                                                                                                 38
    Female                                                                                                                                                           153
    Female                                                                                                                                                             9
    Female (props for making this a freeform field, though)                                                                                                            1
    Female assigned at birth                                                                                                                                           1
    Female or Multi-Gender Femme                                                                                                                                       1
    Fluid                                                                                                                                                              1
    Genderfluid                                                                                                                                                        1
    Genderfluid (born female)                                                                                                                                          1
    Genderflux demi-girl                                                                                                                                               1
    Genderqueer                                                                                                                                                        1
    Human                                                                                                                                                              1
    I identify as female.                                                                                                                                              1
    I'm a man why didn't you make this a drop down question. You should of asked sex? And I would of answered yes please. Seriously how much text can this take?       1
    M                                                                                                                                                                 86
    MALE                                                                                                                                                               1
    Male                                                                                                                                                             610
    Male                                                                                                                                                              11
                                                                                                                                                                    ... 
    Sex is male                                                                                                                                                        1
    Transgender woman                                                                                                                                                  1
    Transitioned, M2F                                                                                                                                                  1
    Unicorn                                                                                                                                                            1
    Woman                                                                                                                                                              3
    cis male                                                                                                                                                           1
    cis man                                                                                                                                                            1
    cisdude                                                                                                                                                            1
    f                                                                                                                                                                 23
    fem                                                                                                                                                                1
    female                                                                                                                                                            95
    female                                                                                                                                                             3
    female-bodied; no feelings about gender                                                                                                                            1
    female/woman                                                                                                                                                       1
    fm                                                                                                                                                                 1
    genderqueer                                                                                                                                                        1
    genderqueer woman                                                                                                                                                  1
    human                                                                                                                                                              1
    m                                                                                                                                                                 79
    mail                                                                                                                                                               1
    male                                                                                                                                                             249
    male                                                                                                                                                               2
    male 9:1 female, roughly                                                                                                                                           1
    man                                                                                                                                                                3
    mtf                                                                                                                                                                1
    nb masculine                                                                                                                                                       1
    non-binary                                                                                                                                                         4
    none of your business                                                                                                                                              1
    woman                                                                                                                                                              4
    NaN                                                                                                                                                                3
    Name: gender, Length: 71, dtype: int64
    --------------frequency table of country_live: ---------------------
    Afghanistan                   2
    Algeria                       1
    Argentina                     1
    Australia                    35
    Austria                       4
    Bangladesh                    1
    Belgium                       5
    Bosnia and Herzegovina        2
    Brazil                       10
    Brunei                        1
    Bulgaria                      7
    Canada                       78
    Chile                         3
    China                         1
    Colombia                      2
    Costa Rica                    1
    Czech Republic                3
    Denmark                       7
    Ecuador                       1
    Estonia                       2
    Finland                       7
    France                       16
    Germany                      58
    Greece                        1
    Guatemala                     1
    Hungary                       1
    India                         9
    Iran                          1
    Ireland                      15
    Israel                        2
    Italy                         5
    Japan                         2
    Lithuania                     2
    Mexico                        2
    Netherlands                  48
    New Zealand                   9
    Norway                        3
    Other                         2
    Pakistan                      3
    Poland                        4
    Romania                       4
    Russia                        9
    Serbia                        1
    Slovakia                      1
    South Africa                  4
    Spain                         4
    Sweden                       19
    Switzerland                  10
    Taiwan                        1
    United Kingdom              180
    United States of America    840
    Venezuela                     1
    Vietnam                       1
    Name: country_live, dtype: int64
    --------------frequency table of state: ---------------------
    Alabama                   4
    Alaska                    2
    Arizona                   5
    California              130
    Colorado                 28
    Connecticut               5
    Delaware                  1
    District of Columbia      3
    Florida                  21
    Georgia                  14
    Idaho                     3
    Illinois                 56
    Indiana                  25
    Iowa                      5
    Kansas                   14
    Kentucky                  4
    Louisiana                 2
    Maine                     5
    Maryland                 16
    Massachusetts            23
    Michigan                 48
    Minnesota                42
    Missouri                 12
    Montana                   1
    Nebraska                 12
    Nevada                    3
    New Hampshire             5
    New Jersey                7
    New Mexico                4
    New York                 45
    North Carolina           21
    North Dakota              4
    Ohio                     25
    Oklahoma                 13
    Oregon                   37
    Pennsylvania             33
    Rhode Island              3
    South Carolina            1
    South Dakota              4
    Tennessee                27
    Texas                    43
    Utah                      6
    Vermont                   5
    Virginia                 15
    Washington               43
    West Virginia             2
    Wisconsin                13
    NaN                     593
    Name: state, dtype: int64
    --------------frequency table of position: ---------------------
    Back-end Developer                                                                                     263
    Back-end Developer|Dev Evangelist/Advocate                                                               2
    Back-end Developer|Dev Evangelist/Advocate|Supervisor/Team Lead                                          2
    Back-end Developer|DevOps/SysAdmin                                                                      16
    Back-end Developer|DevOps/SysAdmin|Dev Evangelist/Advocate|Supervisor/Team Lead                          2
    Back-end Developer|DevOps/SysAdmin|Supervisor/Team Lead                                                  2
    Back-end Developer|DevOps/SysAdmin|Supervisor/Team Lead|Executive Leadership                             1
    Back-end Developer|DevOps/SysAdmin|Supervisor/Team Lead|Other                                            1
    Back-end Developer|Front-end Developer                                                                  61
    Back-end Developer|Front-end Developer|Designer                                                          4
    Back-end Developer|Front-end Developer|One-person shop                                                   3
    Back-end Developer|One-person shop                                                                       7
    Back-end Developer|Supervisor/Team Lead                                                                  4
    Back-end Developer|Supervisor/Team Lead|Other                                                            1
    Back-end Developer|Support|DevOps/SysAdmin                                                               1
    Back-end Developer|Support|Supervisor/Team Lead                                                          1
    Designer                                                                                                28
    Designer|Front-end Developer                                                                             4
    Designer|Front-end Developer|Back-end Developer                                                          1
    Designer|Front-end Developer|Back-end Developer|DevOps/SysAdmin                                          1
    Designer|Front-end Developer|Back-end Developer|DevOps/SysAdmin|Other                                    1
    Designer|Front-end Developer|Back-end Developer|Executive Leadership|Other                               1
    Designer|Front-end Developer|Back-end Developer|Other                                                    1
    Designer|Front-end Developer|Back-end Developer|Sales|Supervisor/Team Lead                               1
    Designer|Front-end Developer|Back-end Developer|Supervisor/Team Lead                                     2
    Designer|Front-end Developer|Back-end Developer|Supervisor/Team Lead|Executive Leadership                1
    Designer|Front-end Developer|Back-end Developer|Support|DevOps/SysAdmin|Supervisor/Team Lead             1
    Designer|One-person shop                                                                                 1
    Designer|Supervisor/Team Lead                                                                            1
    Designer|Support|Supervisor/Team Lead                                                                    1
                                                                                                          ... 
    Supervisor/Team Lead|Executive Leadership                                                                2
    Supervisor/Team Lead|Front-end Developer                                                                 6
    Supervisor/Team Lead|Front-end Developer|Back-end Developer                                              2
    Supervisor/Team Lead|Front-end Developer|Back-end Developer|Dev Evangelist/Advocate                      1
    Supervisor/Team Lead|Front-end Developer|Back-end Developer|DevOps/SysAdmin                              2
    Supervisor/Team Lead|Front-end Developer|Back-end Developer|DevOps/SysAdmin|Dev Evangelist/Advocate      1
    Supervisor/Team Lead|Front-end Developer|Back-end Developer|Support|DevOps/SysAdmin                      1
    Supervisor/Team Lead|Front-end Developer|Designer                                                        1
    Supervisor/Team Lead|Other                                                                               1
    Supervisor/Team Lead|Sales                                                                               1
    Supervisor/Team Lead|Support                                                                             3
    Supervisor/Team Lead|Support|Back-end Developer                                                          2
    Supervisor/Team Lead|Support|Back-end Developer|Front-end Developer                                      1
    Supervisor/Team Lead|Support|Front-end Developer|Back-end Developer                                      1
    Supervisor/Team Lead|Support|One-person shop|Designer|Front-end Developer|Sales                          1
    Support                                                                                                 34
    Support|Back-end Developer                                                                               6
    Support|Back-end Developer|Front-end Developer                                                           4
    Support|Back-end Developer|Front-end Developer|Designer                                                  1
    Support|Back-end Developer|Front-end Developer|One-person shop                                           3
    Support|Back-end Developer|One-person shop                                                               2
    Support|Designer                                                                                         1
    Support|Designer|Front-end Developer                                                                     1
    Support|DevOps/SysAdmin                                                                                  2
    Support|Front-end Developer|Back-end Developer                                                           2
    Support|Front-end Developer|Designer                                                                     1
    Support|HR|Supervisor/Team Lead|Executive Leadership                                                     1
    Support|Other                                                                                            3
    Support|Sales|Back-end Developer|Front-end Developer|Designer|One-person shop                            1
    Support|Sales|Designer                                                                                   1
    Name: position, Length: 264, dtype: int64
    --------------frequency table of remote: ---------------------
    Always       343
    Never        333
    Sometimes    757
    Name: remote, dtype: int64
    

Finally, this `diagnose` variable is the target of interest: whether respondents are diganosed with any mental conditions.


```python
df_s['diagnose'].value_counts()
```




    No     579
    Yes    567
    Name: diagnose, dtype: int64



#### Cleaning: numeric variables

After reviewing the raw frequencies, I noticed that `age` has some extreme values. Let's take a look:


```python
df['age'].mean()
```




    34.28611304954641




```python
sns.set(style="ticks", font_scale=1.2)
ax = sns.boxplot(x=df_s["age"])
sns.despine()
```


![png](/assets/images/post3/output_23_0.png)


Though average age is around 34, thre are some outliers as suggested in the boxplot above. Here we could decide what would be a reasonable range for this numeric variable. For age and for this particular study topic, I would limit the range to `[15:90]`. <br>
Usually when we design surveys and programming the web instrument, we tend to add value checks like this in order to prompt respondents to enter the correct value. For this particular dataset, since I was not given an instrument for the study, it's hard to think from a designer's perspective.

We could take a look at the respondents that said they are either 90+ yrs old and <15 yrs old.


```python
df_s.loc[df_s['age']>90,:]
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
      <th>ID</th>
      <th>diagnose</th>
      <th>employee</th>
      <th>type</th>
      <th>benefit</th>
      <th>discussion</th>
      <th>learn</th>
      <th>anoymity</th>
      <th>leave</th>
      <th>mental_employer</th>
      <th>...</th>
      <th>mental_coworker</th>
      <th>mental_super</th>
      <th>as_serious</th>
      <th>consequence</th>
      <th>age</th>
      <th>gender</th>
      <th>country_live</th>
      <th>state</th>
      <th>position</th>
      <th>remote</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>372</th>
      <td>373</td>
      <td>Yes</td>
      <td>6-25</td>
      <td>1.0</td>
      <td>Yes</td>
      <td>No</td>
      <td>I don't know</td>
      <td>Yes</td>
      <td>Somewhat easy</td>
      <td>No</td>
      <td>...</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>99</td>
      <td>Other</td>
      <td>United States of America</td>
      <td>Michigan</td>
      <td>Supervisor/Team Lead</td>
      <td>Sometimes</td>
    </tr>
    <tr>
      <th>564</th>
      <td>565</td>
      <td>No</td>
      <td>100-500</td>
      <td>1.0</td>
      <td>Yes</td>
      <td>I don't know</td>
      <td>I don't know</td>
      <td>I don't know</td>
      <td>I don't know</td>
      <td>No</td>
      <td>...</td>
      <td>Maybe</td>
      <td>Maybe</td>
      <td>I don't know</td>
      <td>No</td>
      <td>323</td>
      <td>Male</td>
      <td>United States of America</td>
      <td>Oregon</td>
      <td>Back-end Developer</td>
      <td>Sometimes</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 21 columns</p>
</div>




```python
df_s.loc[df_s['age']<15,:]
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
      <th>ID</th>
      <th>diagnose</th>
      <th>employee</th>
      <th>type</th>
      <th>benefit</th>
      <th>discussion</th>
      <th>learn</th>
      <th>anoymity</th>
      <th>leave</th>
      <th>mental_employer</th>
      <th>...</th>
      <th>mental_coworker</th>
      <th>mental_super</th>
      <th>as_serious</th>
      <th>consequence</th>
      <th>age</th>
      <th>gender</th>
      <th>country_live</th>
      <th>state</th>
      <th>position</th>
      <th>remote</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>656</th>
      <td>657</td>
      <td>Yes</td>
      <td>More than 1000</td>
      <td>1.0</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>I don't know</td>
      <td>Somewhat easy</td>
      <td>Maybe</td>
      <td>...</td>
      <td>Maybe</td>
      <td>Yes</td>
      <td>I don't know</td>
      <td>No</td>
      <td>3</td>
      <td>Male</td>
      <td>Canada</td>
      <td>None</td>
      <td>Back-end Developer</td>
      <td>Sometimes</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 21 columns</p>
</div>



In survey research, when we see extreme values like this we would flag these responses and further check for evidence for survey satisficing. This is a when respondents speed through the survey and/or give irresponsible answers. Some well known behaviors include speeding, straightlining, and giving ranodm open-ended answers. It is an interesting methodological topic that worth a separate post.

Here, I consider removing these few records when analyses involve the use of age variable. I'll manually assign `NaN` to the corresponding values.


```python
df_s['age']= df_s['age'].astype('float') # note np.nan is recognized in float type only
df_s['age']= df_s['age'].replace([3], np.nan)
df_s['age']= df_s['age'].replace([99], np.nan)
df_s['age']= df_s['age'].replace([323], np.nan)
```

Though there isn't much change in the average age value, I'm more comfortable now knowing that there were extreme values that I cleaned up and that I should re-inspect the data in terms of response quality.


```python
df_s['age'].mean()
```




    33.37182852143482



#### Cleaning: categorical variables

Again, many things can go under this section as the title implied. Here I'll illustrate a recoding process of a special variable: `country`. <br>
In the survey, the question asked which country respondents currently reside in, and I believe the answer options are standard country names where respondents could select among the list of countries in a drop-down box. 


```python
# remove in the markdown file
df_s['country']= df_s['country_live'].replace(['Other'], 'United States of America')
```


```python
df_s['country'].unique()
```




    array(['United Kingdom', 'United States of America', 'Canada', 'Germany',
           'Netherlands', 'Australia', 'France', 'Belgium', 'Brazil',
           'Denmark', 'Sweden', 'Russia', 'Spain', 'India', 'Mexico',
           'Switzerland', 'Norway', 'Argentina', 'Ireland', 'Italy',
           'Colombia', 'Czech Republic', 'Vietnam', 'Finland', 'Bulgaria',
           'South Africa', 'Slovakia', 'Bangladesh', 'Pakistan',
           'New Zealand', 'Afghanistan', 'Romania', 'Poland', 'Iran',
           'Hungary', 'Israel', 'Japan', 'Ecuador', 'Bosnia and Herzegovina',
           'Austria', 'Chile', 'Estonia'], dtype=object)



Suppose I now want to convert these countries to the continent that they belong to. There is a handy Python [library](https://pypi.org/project/pycountry-convert/) `pycountry-convert` that we could use just for this purpose. <br>
*Step 1: convert `country name` to its standard `country code`:*


```python
country_code = []
for country_name in df_s['country']:
    country_code.append(pc.country_name_to_country_alpha2(country_name))
```


```python
country_code[0:5]
```




    ['GB', 'US', 'GB', 'US', 'GB']



*Step 2: convert `country code` into `continent`:*


```python
continent = []
for code in country_code:
    continent.append(pc.country_alpha2_to_continent_code(code))
```


```python
continent[0:5]
```




    ['EU', 'NA', 'EU', 'NA', 'EU']




```python
df_s['continent'] = continent
df_s['continent'].value_counts()
```




    NA    775
    EU    301
    OC     32
    AS     18
    SA     16
    AF      4
    Name: continent, dtype: int64



Now instead of analysis done on the country level, we have the `continent` variable that's a little more balanced. It's worth noting however, we don't neccessarily have to use `pycountry-convert` to realize the transformation from country name to continent. One could for example just use basic recodes to assign each country to the corresponding continent. This is doable, but I wouldn't recommend - why plant a tree from the beginning when you could stand in the shades of trees other planted?

#### Assign numeric values to the variables

Text values in variables are fine for analysis like frequencies and crosstabs and visualization work, for example, here's a crosstab of `whether remote work always, sometimes, or never` with `ever diagnosed with mental health conditions`:


```python
pd.crosstab(df_s['remote'], df_s['diagnose'])
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
      <th>diagnose</th>
      <th>No</th>
      <th>Yes</th>
    </tr>
    <tr>
      <th>remote</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Always</th>
      <td>116</td>
      <td>101</td>
    </tr>
    <tr>
      <th>Never</th>
      <td>177</td>
      <td>141</td>
    </tr>
    <tr>
      <th>Sometimes</th>
      <td>286</td>
      <td>325</td>
    </tr>
  </tbody>
</table>
</div>



You could imagine that if I were to throw both variables into a logistic regression, it would be hard to work with text values. <br>
In cases like this, we can simply use `.map` function to map the text values to numeric values. `.replace` would also work in this setting.


```python
df_s['remote'] = df_s['remote'].map({'Always': 1, 'Sometimes': 2, 'Never':3})
```


```python
df_s['remote'].unique()
```




    array([2, 3, 1], dtype=int64)


