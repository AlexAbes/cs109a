---
title: Meta-genre mapping
layout: default
nav_include: 1
nav_include: 2
---


```python
import numpy as np
import pandas as pd
import matplotlib
from scipy import stats, integrate
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.api import OLS
import sklearn.metrics as metrics
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.utils import resample
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import Imputer
from sklearn.metrics import mean_squared_error
from datetime import datetime
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler

from sklearn.tree import export_graphviz
from IPython.display import Image
from IPython.display import display
%matplotlib inline
import seaborn.apionly as sns
sns.set_context("poster")
```


    /anaconda/lib/python3.6/site-packages/statsmodels/compat/pandas.py:56: FutureWarning: The pandas.core.datetools module is deprecated and will be removed in a future version. Please use the pandas.tseries module instead.
      from pandas.core import datetools


### Load Data



```python
df = pd.read_csv('playlists_categories_plus_audio_artists_metagenres.csv', encoding='ISO-8859-1')
# For purposes of visualization we use the duration in minutes
df["average_duration_min"] = df["average_duration_ms"] * 1.66667e-5
df["total_duration_min"] = df["total_duration"] * 1.66667e-5
df.head()
```





<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>playlist_name</th>
      <th>playlist_id</th>
      <th>followers</th>
      <th>average_danceability</th>
      <th>average_energy</th>
      <th>average_key</th>
      <th>average_loudness</th>
      <th>average_mode</th>
      <th>average_speechiness</th>
      <th>average_acousticness</th>
      <th>...</th>
      <th>average_tempo</th>
      <th>average_time_signature</th>
      <th>average_duration_ms</th>
      <th>total_duration</th>
      <th>total_num</th>
      <th>genre</th>
      <th>max_popularity</th>
      <th>avg_popularity</th>
      <th>average_duration_min</th>
      <th>total_duration_min</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Today's Top Hits</td>
      <td>37i9dQZF1DXcBWIGoYBM5M</td>
      <td>18098330</td>
      <td>0.671280</td>
      <td>0.655880</td>
      <td>4.980000</td>
      <td>-5.443000</td>
      <td>0.60000</td>
      <td>0.082964</td>
      <td>0.197658</td>
      <td>...</td>
      <td>122.747080</td>
      <td>3.980000</td>
      <td>208131.2200</td>
      <td>10406561</td>
      <td>50</td>
      <td>pop</td>
      <td>97</td>
      <td>80.240000</td>
      <td>3.468861</td>
      <td>173.443030</td>
    </tr>
    <tr>
      <th>1</th>
      <td>RapCaviar</td>
      <td>37i9dQZF1DX0XUsuxWHRQd</td>
      <td>8295257</td>
      <td>0.752765</td>
      <td>0.622176</td>
      <td>5.568627</td>
      <td>-6.668294</td>
      <td>0.54902</td>
      <td>0.234988</td>
      <td>0.178904</td>
      <td>...</td>
      <td>129.420941</td>
      <td>4.039216</td>
      <td>211914.7647</td>
      <td>10807653</td>
      <td>51</td>
      <td>rap</td>
      <td>96</td>
      <td>73.588235</td>
      <td>3.531920</td>
      <td>180.127910</td>
    </tr>
    <tr>
      <th>2</th>
      <td>mint</td>
      <td>37i9dQZF1DX4dyzvuaRJ0n</td>
      <td>4595392</td>
      <td>0.649760</td>
      <td>0.775480</td>
      <td>4.460000</td>
      <td>-5.624800</td>
      <td>0.50000</td>
      <td>0.072348</td>
      <td>0.116406</td>
      <td>...</td>
      <td>125.649840</td>
      <td>3.960000</td>
      <td>211429.6400</td>
      <td>10571482</td>
      <td>50</td>
      <td>electronic</td>
      <td>89</td>
      <td>59.000000</td>
      <td>3.523834</td>
      <td>176.191719</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Are &amp; Be</td>
      <td>37i9dQZF1DX4SBhb3fqCJd</td>
      <td>3777270</td>
      <td>0.627640</td>
      <td>0.538420</td>
      <td>5.280000</td>
      <td>-7.312480</td>
      <td>0.52000</td>
      <td>0.129162</td>
      <td>0.223716</td>
      <td>...</td>
      <td>117.075600</td>
      <td>3.980000</td>
      <td>224363.6000</td>
      <td>11218180</td>
      <td>50</td>
      <td>pop</td>
      <td>87</td>
      <td>64.080000</td>
      <td>3.739401</td>
      <td>186.970041</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Rock This</td>
      <td>37i9dQZF1DXcF6B6QPhFDv</td>
      <td>3992066</td>
      <td>0.526160</td>
      <td>0.787100</td>
      <td>5.000000</td>
      <td>-5.252220</td>
      <td>0.56000</td>
      <td>0.064722</td>
      <td>0.050970</td>
      <td>...</td>
      <td>126.461040</td>
      <td>3.900000</td>
      <td>220319.1800</td>
      <td>11015959</td>
      <td>50</td>
      <td>rock</td>
      <td>74</td>
      <td>60.260000</td>
      <td>3.671994</td>
      <td>183.599684</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 23 columns</p>
</div>



#### The next two visualizations shows the frequency and distribution of playlist durations and song durations in our dataset. This is useful to know before making any decisions on how we might want to manipulate the data because we can see how it is skewed and get a better sense of common trends.



```python
#distribution of the total duration of playlists (milliseconds)
#sns.set(color_codes=True)
sns.distplot(df["total_duration_min"],rug=True, hist_kws = {'alpha': 0.25, 'label': 'Playlist Duration'});
plt.legend();
```



![png](Final%2BProject%2BEDA_files/Final%2BProject%2BEDA_4_0.png)




```python
#distribution of the avg. duration of songs (milliseconds)
sns.distplot(df["average_duration_min"],rug=True, hist_kws = {'alpha': 0.25, 'label': 'Avg. Song Duration'});
plt.legend();
```



![png](Final%2BProject%2BEDA_files/Final%2BProject%2BEDA_5_0.png)


#### There are significant outliers in the dataset that we need to be aware of for the number of end followers; followers does not follow a normal distribution in the dataset, and instead it is skewed with a right tail. What can hardly be seen here is that there are just a few outliers out to the right. We will need to decide whether to include these outliers or not in the data we eventually use.




```python
plt.hist(df["followers"])
plt.ylabel("Frequency")
plt.xlabel("Followers (binned)");
```



![png](Final%2BProject%2BEDA_files/Final%2BProject%2BEDA_7_0.png)


#### All average audio information variables (seen in the visualizations below) appear to have meaningful relationships with the response variable; some more than others, but some are linear and some (such as mode) appear to be curved.



```python
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.scatter(df["average_danceability"], df["followers"])
ax1.set_ylabel('Followers')
ax1.set_xlabel('Avg. Danceability')
ax2.scatter(df["average_energy"], df["followers"])
ax2.set_xlabel('Avg. Energy')

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.scatter(df["average_mode"], df["followers"])
ax1.set_ylabel('Followers')
ax1.set_xlabel('Avg. Mode')
ax2.scatter(df["average_loudness"], df["followers"])
ax2.set_xlabel('Avg. Loudness')

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.scatter(df["average_speechiness"], df["followers"])
ax1.set_ylabel('Followers')
ax1.set_xlabel('Avg. Speechiness')
ax2.scatter(df["average_acousticness"], df["followers"])
ax2.set_xlabel('Avg. Acousticness')


```





    <matplotlib.text.Text at 0x11a487be0>




![png](Final%2BProject%2BEDA_files/Final%2BProject%2BEDA_9_1.png)



![png](Final%2BProject%2BEDA_files/Final%2BProject%2BEDA_9_2.png)



![png](Final%2BProject%2BEDA_files/Final%2BProject%2BEDA_9_3.png)




```python
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.scatter(df["average_instrumentalness"], df["followers"])
ax1.set_ylabel('Followers')
ax1.set_xlabel('Avg. Instrumentalness')
ax2.scatter(df["average_liveness"], df["followers"])
ax2.set_xlabel('Avg. Liveness')

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.scatter(df["average_valence"], df["followers"])
ax1.set_ylabel('Followers')
ax1.set_xlabel('Avg. Valence')
ax2.scatter(df["average_tempo"], df["followers"])
ax2.set_xlabel('Avg. Tempo')

f, (ax1) = plt.subplots(1, 1, sharey=True)
ax1.scatter(df["average_time_signature"], df["followers"])
ax1.set_ylabel('Followers')
ax1.set_xlabel('Avg. Speechiness')

```





    <matplotlib.text.Text at 0x11b2f0dd8>




![png](Final%2BProject%2BEDA_files/Final%2BProject%2BEDA_10_1.png)



![png](Final%2BProject%2BEDA_files/Final%2BProject%2BEDA_10_2.png)



![png](Final%2BProject%2BEDA_files/Final%2BProject%2BEDA_10_3.png)


#### From the below visualizationa it appears that average song duration and total playlist duration are both highly significant variables. Both appear to have optimal values peaking at smaller numbers, rather than a strictly linear relationship. Therefore we will likely attempt modeling these terms quadratically in some way.




```python
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.scatter(df["average_duration_ms"], df["followers"])
ax1.set_ylabel('Followers')
ax1.set_xlabel('Avg. Song Duration')
ax2.scatter(df["total_duration"], df["followers"])
ax2.set_xlabel('Playlist Duration')

f, (ax1) = plt.subplots(1, 1, sharey=True)
ax1.scatter(df["total_num"], df["followers"])
ax1.set_ylabel('Followers')
ax1.set_xlabel('Number of Songs')
```





    <matplotlib.text.Text at 0x11be067b8>




![png](Final%2BProject%2BEDA_files/Final%2BProject%2BEDA_12_1.png)



![png](Final%2BProject%2BEDA_files/Final%2BProject%2BEDA_12_2.png)


#### A quick test for colinearity provides us with good information about the information that our predictors encode not just individually, but as an overall dataset. We wanted to look at these relationships as we will need to be aware of what predictors are the most powerful and which can be encapsulated by another variable.



```python
sns.set_context("paper")
from pandas.tools.plotting import scatter_matrix
fig, ax = plt.subplots(1,1, figsize=(15,10))
axes = scatter_matrix(df[["average_danceability","average_energy","average_loudness","average_valence",
                          "average_tempo","average_time_signature"]], alpha=0.5, diagonal='kde', ax=ax)
plt.show()
```


    /anaconda/lib/python3.6/site-packages/ipykernel_launcher.py:5: FutureWarning: 'pandas.tools.plotting.scatter_matrix' is deprecated, import 'pandas.plotting.scatter_matrix' instead.
      """
    /anaconda/lib/python3.6/site-packages/ipykernel_launcher.py:5: UserWarning: To output multiple subplots, the figure containing the passed axes is being cleared
      """



![png](Final%2BProject%2BEDA_files/Final%2BProject%2BEDA_14_1.png)


#### Since we will be trying to create an optimal playlist queried by genre, we decided to catgorize our playlists into 23 genres that described the artists on the tracks included in the playlist. Spotify API did not assign genres to actual playlists so we imputed them through the artist and generalized so that we would have a significant amount of playlists in each genre.



```python
genre_info = ['average_danceability','average_energy','average_key','average_loudness',
              'average_mode','average_speechiness','average_acousticness','average_instrumentalness',
              'average_liveness','average_valence','average_tempo','average_time_signature',
              'average_duration_ms','total_duration','max_popularity','avg_popularity','average_duration_min',
              'total_duration_min']
df_cats_minus_audio = df.drop(genre_info, axis=1)
df_cats_minus_audio.head()
```





<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>playlist_name</th>
      <th>playlist_id</th>
      <th>followers</th>
      <th>total_num</th>
      <th>genre</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Today's Top Hits</td>
      <td>37i9dQZF1DXcBWIGoYBM5M</td>
      <td>18098330</td>
      <td>50</td>
      <td>pop</td>
    </tr>
    <tr>
      <th>1</th>
      <td>RapCaviar</td>
      <td>37i9dQZF1DX0XUsuxWHRQd</td>
      <td>8295257</td>
      <td>51</td>
      <td>rap</td>
    </tr>
    <tr>
      <th>2</th>
      <td>mint</td>
      <td>37i9dQZF1DX4dyzvuaRJ0n</td>
      <td>4595392</td>
      <td>50</td>
      <td>electronic</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Are &amp; Be</td>
      <td>37i9dQZF1DX4SBhb3fqCJd</td>
      <td>3777270</td>
      <td>50</td>
      <td>pop</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Rock This</td>
      <td>37i9dQZF1DXcF6B6QPhFDv</td>
      <td>3992066</td>
      <td>50</td>
      <td>rock</td>
    </tr>
  </tbody>
</table>
</div>



#### The below visualization demonstrates that there are significant differences in the mean number of followers between playlists of different genres. Therefore we will likely sacrifice the benefits of a larger dataset for the usefulness of the added genre information in the smaller dataset. Some genres do not have enough playlists in them, e.g. Comedy has only 2 playlists. We will likely have to cull this category from the dataset. In addition, we may wish to model ‘speech-y’ categories such literature and discussion differently from musical playlists, since their length is longer and they are generally less popular.




```python
# bar chart of categories versus followers
# drop the None genre
df_cats_minus_audio = df_cats_minus_audio.drop(df_cats_minus_audio[df_cats_minus_audio.genre == "None"].index)
g_object = df_cats_minus_audio.groupby("genre").mean()
sns.set_context("poster")
g = sns.barplot(x='genre', y='followers', data=df_cats_minus_audio)
plt.xticks(rotation=90);
```



![png](Final%2BProject%2BEDA_files/Final%2BProject%2BEDA_18_0.png)
