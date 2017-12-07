---
title: EDA
notebook: Final+Project+EDA.ipynb
nav_include: 1
nav_include: 2
nav_include: 3
nav_include: 4
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
```


    /anaconda/lib/python3.6/site-packages/statsmodels/compat/pandas.py:56: FutureWarning: The pandas.core.datetools module is deprecated and will be removed in a future version. Please use the pandas.tseries module instead.
      from pandas.core import datetools


### Load Data



```python
df = pd.read_csv('playlists_categories_plus_audio_artists_metagenres.csv', encoding='ISO-8859-1')
df["average_duration_min"] = df["average_duration_ms"] * 1.66667e-5
df["total_duration_min"] = df["total_duration"] * 1.66667e-5
df
df
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
      <td>0.600000</td>
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
      <td>0.549020</td>
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
      <td>0.500000</td>
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
      <td>0.520000</td>
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
      <td>0.560000</td>
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
    <tr>
      <th>5</th>
      <td>Hot Country</td>
      <td>37i9dQZF1DX1lVhptIYRda</td>
      <td>4244887</td>
      <td>0.586333</td>
      <td>0.718255</td>
      <td>6.058824</td>
      <td>-5.756765</td>
      <td>0.901961</td>
      <td>0.043808</td>
      <td>0.159697</td>
      <td>...</td>
      <td>118.366275</td>
      <td>3.980392</td>
      <td>195283.8039</td>
      <td>9959474</td>
      <td>51</td>
      <td>country</td>
      <td>90</td>
      <td>66.764706</td>
      <td>3.254737</td>
      <td>165.991565</td>
    </tr>
    <tr>
      <th>6</th>
      <td>åÁViva Latino!</td>
      <td>37i9dQZF1DX10zKzsJ2jva</td>
      <td>6607264</td>
      <td>0.729380</td>
      <td>0.768520</td>
      <td>4.980000</td>
      <td>-4.832860</td>
      <td>0.620000</td>
      <td>0.088064</td>
      <td>0.165950</td>
      <td>...</td>
      <td>119.758800</td>
      <td>4.000000</td>
      <td>217256.6400</td>
      <td>10862832</td>
      <td>50</td>
      <td>world</td>
      <td>91</td>
      <td>79.480000</td>
      <td>3.620951</td>
      <td>181.047562</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Afternoon Acoustic</td>
      <td>37i9dQZF1DX4E3UdUs7fUx</td>
      <td>2470127</td>
      <td>0.525704</td>
      <td>0.325552</td>
      <td>5.370370</td>
      <td>-11.725062</td>
      <td>0.777778</td>
      <td>0.033222</td>
      <td>0.786123</td>
      <td>...</td>
      <td>114.429407</td>
      <td>3.913580</td>
      <td>228501.9506</td>
      <td>18508658</td>
      <td>81</td>
      <td>indie</td>
      <td>75</td>
      <td>51.246914</td>
      <td>3.808373</td>
      <td>308.478250</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Peaceful Piano</td>
      <td>37i9dQZF1DX4sWSpwq3LiO</td>
      <td>3313734</td>
      <td>0.232568</td>
      <td>0.041348</td>
      <td>3.006173</td>
      <td>-15.045525</td>
      <td>0.277778</td>
      <td>0.028737</td>
      <td>0.609469</td>
      <td>...</td>
      <td>66.667210</td>
      <td>2.191358</td>
      <td>109436.0926</td>
      <td>17728647</td>
      <td>162</td>
      <td>ambient</td>
      <td>74</td>
      <td>36.759259</td>
      <td>1.823939</td>
      <td>295.478041</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Roots Rising</td>
      <td>37i9dQZF1DWYV7OOaGhoH0</td>
      <td>987430</td>
      <td>0.491962</td>
      <td>0.427081</td>
      <td>5.238095</td>
      <td>-9.009171</td>
      <td>0.685714</td>
      <td>0.035198</td>
      <td>0.533639</td>
      <td>...</td>
      <td>109.990086</td>
      <td>3.676190</td>
      <td>221428.0000</td>
      <td>23249940</td>
      <td>105</td>
      <td>indie</td>
      <td>77</td>
      <td>48.228571</td>
      <td>3.690474</td>
      <td>387.499775</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Lost In Austin: Country Music from SXSW</td>
      <td>5JdIoXkjfuyWmDdDnznmq1</td>
      <td>3822</td>
      <td>0.502805</td>
      <td>0.573552</td>
      <td>5.317073</td>
      <td>-8.520195</td>
      <td>0.902439</td>
      <td>0.047393</td>
      <td>0.372465</td>
      <td>...</td>
      <td>128.784439</td>
      <td>3.902439</td>
      <td>216742.0610</td>
      <td>17772849</td>
      <td>82</td>
      <td>None</td>
      <td>76</td>
      <td>21.524390</td>
      <td>3.612375</td>
      <td>296.214742</td>
    </tr>
    <tr>
      <th>11</th>
      <td>SXSW: HipHopUpNext</td>
      <td>37i9dQZF1DX5PuwKY2VZBZ</td>
      <td>14524</td>
      <td>0.741000</td>
      <td>0.591960</td>
      <td>5.040000</td>
      <td>-7.494200</td>
      <td>0.360000</td>
      <td>0.188500</td>
      <td>0.220964</td>
      <td>...</td>
      <td>124.331680</td>
      <td>3.880000</td>
      <td>199848.6800</td>
      <td>4996217</td>
      <td>25</td>
      <td>rap</td>
      <td>68</td>
      <td>42.400000</td>
      <td>3.330818</td>
      <td>83.270450</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Women of SXSW</td>
      <td>37i9dQZF1DXboGLA1CNczK</td>
      <td>3352</td>
      <td>0.552673</td>
      <td>0.567402</td>
      <td>4.872727</td>
      <td>-7.783618</td>
      <td>0.690909</td>
      <td>0.072133</td>
      <td>0.344474</td>
      <td>...</td>
      <td>121.398164</td>
      <td>3.872727</td>
      <td>218424.1818</td>
      <td>12013330</td>
      <td>55</td>
      <td>None</td>
      <td>59</td>
      <td>30.854545</td>
      <td>3.640410</td>
      <td>200.222567</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Music Inspired By Mogul</td>
      <td>4HghCZXykAY0kdRqJipQJd</td>
      <td>3638</td>
      <td>0.807476</td>
      <td>0.628619</td>
      <td>5.904762</td>
      <td>-8.145238</td>
      <td>0.666667</td>
      <td>0.235300</td>
      <td>0.143884</td>
      <td>...</td>
      <td>103.087476</td>
      <td>4.000000</td>
      <td>234515.6667</td>
      <td>4924829</td>
      <td>21</td>
      <td>rap</td>
      <td>72</td>
      <td>48.238095</td>
      <td>3.908602</td>
      <td>82.080647</td>
    </tr>
    <tr>
      <th>14</th>
      <td>SXSW 2017: Daily Guide</td>
      <td>1LYK2ahDnbLnS4fddCFY7z</td>
      <td>7879</td>
      <td>0.573109</td>
      <td>0.667089</td>
      <td>5.581818</td>
      <td>-6.940618</td>
      <td>0.636364</td>
      <td>0.097662</td>
      <td>0.218213</td>
      <td>...</td>
      <td>121.318527</td>
      <td>3.854545</td>
      <td>219494.6182</td>
      <td>12072204</td>
      <td>55</td>
      <td>rock</td>
      <td>82</td>
      <td>42.436364</td>
      <td>3.658251</td>
      <td>201.203802</td>
    </tr>
    <tr>
      <th>15</th>
      <td>TGIF</td>
      <td>37i9dQZF1DXcRXFNfZr7Tp</td>
      <td>1454537</td>
      <td>0.698060</td>
      <td>0.653340</td>
      <td>5.300000</td>
      <td>-5.828860</td>
      <td>0.480000</td>
      <td>0.086456</td>
      <td>0.181231</td>
      <td>...</td>
      <td>112.946660</td>
      <td>4.000000</td>
      <td>202996.2000</td>
      <td>10149810</td>
      <td>50</td>
      <td>electronic</td>
      <td>96</td>
      <td>72.860000</td>
      <td>3.383277</td>
      <td>169.163838</td>
    </tr>
    <tr>
      <th>16</th>
      <td>SXSW UK Rising</td>
      <td>37i9dQZF1DXdIcOzOLoAn6</td>
      <td>8186</td>
      <td>0.540706</td>
      <td>0.692353</td>
      <td>4.313725</td>
      <td>-7.438020</td>
      <td>0.725490</td>
      <td>0.098094</td>
      <td>0.200536</td>
      <td>...</td>
      <td>129.127647</td>
      <td>3.980392</td>
      <td>214371.7451</td>
      <td>10932959</td>
      <td>51</td>
      <td>None</td>
      <td>67</td>
      <td>29.294118</td>
      <td>3.572870</td>
      <td>182.216348</td>
    </tr>
    <tr>
      <th>17</th>
      <td>ExperiWOMENtell</td>
      <td>37i9dQZF1DWY7uXxMycXfq</td>
      <td>1726</td>
      <td>0.297418</td>
      <td>0.337120</td>
      <td>4.425000</td>
      <td>-15.395775</td>
      <td>0.525000</td>
      <td>0.049460</td>
      <td>0.776473</td>
      <td>...</td>
      <td>108.700025</td>
      <td>3.875000</td>
      <td>332391.0750</td>
      <td>13295643</td>
      <td>40</td>
      <td>other</td>
      <td>51</td>
      <td>18.500000</td>
      <td>5.539862</td>
      <td>221.594493</td>
    </tr>
    <tr>
      <th>18</th>
      <td>The Refugee Playlist</td>
      <td>37i9dQZF1DX1328t2iygZy</td>
      <td>16047</td>
      <td>0.606864</td>
      <td>0.543186</td>
      <td>3.181818</td>
      <td>-9.109955</td>
      <td>0.818182</td>
      <td>0.088873</td>
      <td>0.348700</td>
      <td>...</td>
      <td>110.400409</td>
      <td>3.818182</td>
      <td>227756.5909</td>
      <td>5010645</td>
      <td>22</td>
      <td>pop</td>
      <td>67</td>
      <td>42.681818</td>
      <td>3.795951</td>
      <td>83.510917</td>
    </tr>
    <tr>
      <th>19</th>
      <td>This Is: Jason Isbell</td>
      <td>37i9dQZF1DX1wvY0jqbS55</td>
      <td>11349</td>
      <td>0.490653</td>
      <td>0.441812</td>
      <td>5.673469</td>
      <td>-10.937306</td>
      <td>0.816327</td>
      <td>0.033718</td>
      <td>0.361513</td>
      <td>...</td>
      <td>112.118735</td>
      <td>3.897959</td>
      <td>249612.0612</td>
      <td>12230991</td>
      <td>49</td>
      <td>country</td>
      <td>64</td>
      <td>43.448980</td>
      <td>4.160209</td>
      <td>203.850258</td>
    </tr>
    <tr>
      <th>20</th>
      <td>This Is: Disney</td>
      <td>37i9dQZF1DWVs8I62NcHks</td>
      <td>247039</td>
      <td>0.471172</td>
      <td>0.434197</td>
      <td>5.017241</td>
      <td>-10.861914</td>
      <td>0.862069</td>
      <td>0.075200</td>
      <td>0.607912</td>
      <td>...</td>
      <td>110.555552</td>
      <td>3.862069</td>
      <td>161395.8621</td>
      <td>9360960</td>
      <td>58</td>
      <td>soundtrack</td>
      <td>73</td>
      <td>41.137931</td>
      <td>2.689936</td>
      <td>156.016312</td>
    </tr>
    <tr>
      <th>21</th>
      <td>This Is: The Doors</td>
      <td>37i9dQZF1DX95gx8SY6DLX</td>
      <td>82684</td>
      <td>0.556000</td>
      <td>0.514140</td>
      <td>5.180000</td>
      <td>-11.111840</td>
      <td>0.600000</td>
      <td>0.045362</td>
      <td>0.350730</td>
      <td>...</td>
      <td>119.263040</td>
      <td>3.980000</td>
      <td>255543.7600</td>
      <td>12777188</td>
      <td>50</td>
      <td>rock</td>
      <td>70</td>
      <td>51.280000</td>
      <td>4.259071</td>
      <td>212.953559</td>
    </tr>
    <tr>
      <th>22</th>
      <td>This Is: AC/DC</td>
      <td>37i9dQZF1DXec50AjHrNTq</td>
      <td>522221</td>
      <td>0.508886</td>
      <td>0.849000</td>
      <td>6.657143</td>
      <td>-4.677229</td>
      <td>0.828571</td>
      <td>0.082583</td>
      <td>0.043497</td>
      <td>...</td>
      <td>126.481629</td>
      <td>4.000000</td>
      <td>249849.8857</td>
      <td>8744746</td>
      <td>35</td>
      <td>rock</td>
      <td>80</td>
      <td>61.457143</td>
      <td>4.164173</td>
      <td>145.746058</td>
    </tr>
    <tr>
      <th>23</th>
      <td>This Is: Adele</td>
      <td>37i9dQZF1DWZUozJiHy44Y</td>
      <td>527006</td>
      <td>0.521480</td>
      <td>0.479440</td>
      <td>4.800000</td>
      <td>-6.989600</td>
      <td>0.480000</td>
      <td>0.039696</td>
      <td>0.464827</td>
      <td>...</td>
      <td>120.981760</td>
      <td>3.760000</td>
      <td>254334.0000</td>
      <td>6358350</td>
      <td>25</td>
      <td>pop</td>
      <td>77</td>
      <td>63.840000</td>
      <td>4.238908</td>
      <td>105.972712</td>
    </tr>
    <tr>
      <th>24</th>
      <td>This Is: Al Green</td>
      <td>37i9dQZF1DWZMCPjHG57gq</td>
      <td>27122</td>
      <td>0.663500</td>
      <td>0.431000</td>
      <td>5.269231</td>
      <td>-11.585269</td>
      <td>0.807692</td>
      <td>0.064419</td>
      <td>0.330031</td>
      <td>...</td>
      <td>114.878615</td>
      <td>3.846154</td>
      <td>235187.1154</td>
      <td>6114865</td>
      <td>26</td>
      <td>rock</td>
      <td>73</td>
      <td>22.038462</td>
      <td>3.919793</td>
      <td>101.914620</td>
    </tr>
    <tr>
      <th>25</th>
      <td>This Is: alt-J</td>
      <td>37i9dQZF1DWXMmak2OV7PN</td>
      <td>62309</td>
      <td>0.482478</td>
      <td>0.482261</td>
      <td>5.217391</td>
      <td>-10.758565</td>
      <td>0.782609</td>
      <td>0.041061</td>
      <td>0.524209</td>
      <td>...</td>
      <td>126.467348</td>
      <td>4.000000</td>
      <td>244150.1304</td>
      <td>5615453</td>
      <td>23</td>
      <td>rock</td>
      <td>71</td>
      <td>52.956522</td>
      <td>4.069177</td>
      <td>93.591071</td>
    </tr>
    <tr>
      <th>26</th>
      <td>This Is: Aqua</td>
      <td>37i9dQZF1DX3reMgdkcDa4</td>
      <td>7346</td>
      <td>0.703190</td>
      <td>0.833000</td>
      <td>3.761905</td>
      <td>-5.995667</td>
      <td>0.523810</td>
      <td>0.048476</td>
      <td>0.041955</td>
      <td>...</td>
      <td>129.197191</td>
      <td>4.000000</td>
      <td>220235.4762</td>
      <td>4624945</td>
      <td>21</td>
      <td>dance</td>
      <td>56</td>
      <td>32.428571</td>
      <td>3.670599</td>
      <td>77.082571</td>
    </tr>
    <tr>
      <th>27</th>
      <td>This is: Aretha Franklin</td>
      <td>37i9dQZF1DX6bJVMtDYJHx</td>
      <td>26987</td>
      <td>0.586380</td>
      <td>0.552080</td>
      <td>5.120000</td>
      <td>-9.596640</td>
      <td>0.800000</td>
      <td>0.060648</td>
      <td>0.375784</td>
      <td>...</td>
      <td>118.037160</td>
      <td>3.800000</td>
      <td>237206.8000</td>
      <td>11860340</td>
      <td>50</td>
      <td>oldies</td>
      <td>67</td>
      <td>42.280000</td>
      <td>3.953455</td>
      <td>197.672729</td>
    </tr>
    <tr>
      <th>28</th>
      <td>This Is: Bee Gees</td>
      <td>37i9dQZF1DWZvGllBOMzci</td>
      <td>118382</td>
      <td>0.575231</td>
      <td>0.491448</td>
      <td>5.096154</td>
      <td>-11.453077</td>
      <td>0.673077</td>
      <td>0.041063</td>
      <td>0.267940</td>
      <td>...</td>
      <td>108.464192</td>
      <td>3.980769</td>
      <td>239702.3269</td>
      <td>12464521</td>
      <td>52</td>
      <td>rock</td>
      <td>58</td>
      <td>41.269231</td>
      <td>3.995047</td>
      <td>207.742432</td>
    </tr>
    <tr>
      <th>29</th>
      <td>This Is: blink-182</td>
      <td>37i9dQZF1DX4EKARgqYPFZ</td>
      <td>130409</td>
      <td>0.431185</td>
      <td>0.918292</td>
      <td>5.861538</td>
      <td>-4.953215</td>
      <td>0.953846</td>
      <td>0.083649</td>
      <td>0.010680</td>
      <td>...</td>
      <td>130.107031</td>
      <td>3.923077</td>
      <td>177980.9231</td>
      <td>11568760</td>
      <td>65</td>
      <td>christmas</td>
      <td>73</td>
      <td>39.861538</td>
      <td>2.966355</td>
      <td>192.813052</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1562</th>
      <td>88 Keys</td>
      <td>37i9dQZF1DX561TxkFttR4</td>
      <td>220709</td>
      <td>0.364159</td>
      <td>0.105207</td>
      <td>5.045455</td>
      <td>-23.092307</td>
      <td>0.500000</td>
      <td>0.055019</td>
      <td>0.974420</td>
      <td>...</td>
      <td>110.731954</td>
      <td>3.659091</td>
      <td>224650.7386</td>
      <td>19769265</td>
      <td>88</td>
      <td>ambient</td>
      <td>74</td>
      <td>43.431818</td>
      <td>3.744186</td>
      <td>329.488409</td>
    </tr>
    <tr>
      <th>1563</th>
      <td>Raw Expression</td>
      <td>3yUO32lGCEnAfFGORSEsy4</td>
      <td>0</td>
      <td>0.725976</td>
      <td>0.778048</td>
      <td>4.404762</td>
      <td>-5.068429</td>
      <td>0.642857</td>
      <td>0.241783</td>
      <td>0.153471</td>
      <td>...</td>
      <td>116.585286</td>
      <td>4.119048</td>
      <td>185197.3095</td>
      <td>7778287</td>
      <td>42</td>
      <td>dance</td>
      <td>49</td>
      <td>20.785714</td>
      <td>3.086628</td>
      <td>129.638376</td>
    </tr>
    <tr>
      <th>1564</th>
      <td>Sunshine Reggae</td>
      <td>37i9dQZF1DXbwoaqxaoAVr</td>
      <td>159270</td>
      <td>0.688360</td>
      <td>0.683400</td>
      <td>5.340000</td>
      <td>-6.124680</td>
      <td>0.520000</td>
      <td>0.089900</td>
      <td>0.118310</td>
      <td>...</td>
      <td>112.293920</td>
      <td>4.000000</td>
      <td>221566.6400</td>
      <td>11078332</td>
      <td>50</td>
      <td>pop</td>
      <td>82</td>
      <td>39.860000</td>
      <td>3.692785</td>
      <td>184.639236</td>
    </tr>
    <tr>
      <th>1565</th>
      <td>Morning Rhythm</td>
      <td>37i9dQZF1DX3ohNxI5tB79</td>
      <td>115660</td>
      <td>0.668420</td>
      <td>0.673880</td>
      <td>5.500000</td>
      <td>-6.902540</td>
      <td>0.440000</td>
      <td>0.090506</td>
      <td>0.153403</td>
      <td>...</td>
      <td>110.387340</td>
      <td>3.980000</td>
      <td>235914.6600</td>
      <td>11795733</td>
      <td>50</td>
      <td>indie</td>
      <td>78</td>
      <td>50.640000</td>
      <td>3.931919</td>
      <td>196.595943</td>
    </tr>
    <tr>
      <th>1566</th>
      <td>I Love My Neo Soul</td>
      <td>37i9dQZF1DX44dZ4p5QLf4</td>
      <td>96655</td>
      <td>0.685143</td>
      <td>0.552041</td>
      <td>6.061224</td>
      <td>-7.830673</td>
      <td>0.448980</td>
      <td>0.166153</td>
      <td>0.309949</td>
      <td>...</td>
      <td>101.375898</td>
      <td>3.959184</td>
      <td>251849.5918</td>
      <td>12340630</td>
      <td>49</td>
      <td>soul</td>
      <td>61</td>
      <td>32.836735</td>
      <td>4.197502</td>
      <td>205.677578</td>
    </tr>
    <tr>
      <th>1567</th>
      <td>Afternoon Train Ride</td>
      <td>37i9dQZF1DX8MbMfAHb8U0</td>
      <td>25081</td>
      <td>0.670477</td>
      <td>0.520277</td>
      <td>5.584615</td>
      <td>-8.737508</td>
      <td>0.415385</td>
      <td>0.130618</td>
      <td>0.354701</td>
      <td>...</td>
      <td>113.949446</td>
      <td>3.923077</td>
      <td>245581.1846</td>
      <td>15962777</td>
      <td>65</td>
      <td>soul</td>
      <td>60</td>
      <td>24.646154</td>
      <td>4.093028</td>
      <td>266.046815</td>
    </tr>
    <tr>
      <th>1568</th>
      <td>The Midnight Hour</td>
      <td>37i9dQZF1DXea80XwOJRgD</td>
      <td>50982</td>
      <td>0.576022</td>
      <td>0.504391</td>
      <td>6.108696</td>
      <td>-7.714065</td>
      <td>0.586957</td>
      <td>0.048130</td>
      <td>0.358605</td>
      <td>...</td>
      <td>121.991674</td>
      <td>3.543478</td>
      <td>235391.3696</td>
      <td>10828003</td>
      <td>46</td>
      <td>blues</td>
      <td>79</td>
      <td>37.913043</td>
      <td>3.923197</td>
      <td>180.467078</td>
    </tr>
    <tr>
      <th>1569</th>
      <td>Soultronic</td>
      <td>37i9dQZF1DXdc0DUqaW3MZ</td>
      <td>99996</td>
      <td>0.655800</td>
      <td>0.567380</td>
      <td>5.660000</td>
      <td>-8.181940</td>
      <td>0.360000</td>
      <td>0.137518</td>
      <td>0.236202</td>
      <td>...</td>
      <td>115.374080</td>
      <td>3.980000</td>
      <td>217142.4800</td>
      <td>10857124</td>
      <td>50</td>
      <td>None</td>
      <td>74</td>
      <td>44.760000</td>
      <td>3.619049</td>
      <td>180.952429</td>
    </tr>
    <tr>
      <th>1570</th>
      <td>The Cookout</td>
      <td>37i9dQZF1DXab8DipvnuNU</td>
      <td>141749</td>
      <td>0.751340</td>
      <td>0.645404</td>
      <td>5.702128</td>
      <td>-8.423000</td>
      <td>0.595745</td>
      <td>0.115126</td>
      <td>0.164604</td>
      <td>...</td>
      <td>106.003532</td>
      <td>4.000000</td>
      <td>262786.9362</td>
      <td>12350986</td>
      <td>47</td>
      <td>blues</td>
      <td>87</td>
      <td>52.021277</td>
      <td>4.379791</td>
      <td>205.850178</td>
    </tr>
    <tr>
      <th>1571</th>
      <td>Icona Pops guide till festivalpackningen</td>
      <td>37i9dQZF1DXdwuuZvpX2qk</td>
      <td>21</td>
      <td>0.657208</td>
      <td>0.561417</td>
      <td>4.541667</td>
      <td>-7.735083</td>
      <td>0.833333</td>
      <td>0.477221</td>
      <td>0.369585</td>
      <td>...</td>
      <td>109.278083</td>
      <td>3.833333</td>
      <td>134045.2083</td>
      <td>3217085</td>
      <td>24</td>
      <td>None</td>
      <td>60</td>
      <td>16.333333</td>
      <td>2.234091</td>
      <td>53.618191</td>
    </tr>
    <tr>
      <th>1572</th>
      <td>MidsommarvÌ_rd: Tomas Ledin</td>
      <td>37i9dQZF1DX4uFp6ryXvMm</td>
      <td>638</td>
      <td>0.611833</td>
      <td>0.556733</td>
      <td>5.566667</td>
      <td>-12.378933</td>
      <td>0.833333</td>
      <td>0.499230</td>
      <td>0.344662</td>
      <td>...</td>
      <td>118.165900</td>
      <td>3.700000</td>
      <td>158580.3333</td>
      <td>4757410</td>
      <td>30</td>
      <td>None</td>
      <td>78</td>
      <td>10.766667</td>
      <td>2.643011</td>
      <td>79.290325</td>
    </tr>
    <tr>
      <th>1573</th>
      <td>Intervju med Seinabo Sey</td>
      <td>2k9EArKbf7N3QUmuNJHSo8</td>
      <td>2</td>
      <td>0.580333</td>
      <td>0.362667</td>
      <td>6.333333</td>
      <td>-11.662333</td>
      <td>1.000000</td>
      <td>0.340000</td>
      <td>0.557667</td>
      <td>...</td>
      <td>112.274667</td>
      <td>4.000000</td>
      <td>277483.3333</td>
      <td>832450</td>
      <td>3</td>
      <td>pop</td>
      <td>53</td>
      <td>20.000000</td>
      <td>4.624731</td>
      <td>13.874194</td>
    </tr>
    <tr>
      <th>1574</th>
      <td>Intervju med Mando Diao</td>
      <td>37i9dQZF1DWZHJcG9TkzIK</td>
      <td>81</td>
      <td>0.632067</td>
      <td>0.556200</td>
      <td>5.533333</td>
      <td>-9.398667</td>
      <td>0.466667</td>
      <td>0.338100</td>
      <td>0.194706</td>
      <td>...</td>
      <td>127.770467</td>
      <td>3.800000</td>
      <td>257472.2667</td>
      <td>3862084</td>
      <td>15</td>
      <td>rock</td>
      <td>39</td>
      <td>14.666667</td>
      <td>4.291213</td>
      <td>64.368195</td>
    </tr>
    <tr>
      <th>1575</th>
      <td>Intervju med Agnes</td>
      <td>37i9dQZF1DX8uJwKSyPfRr</td>
      <td>104</td>
      <td>0.585941</td>
      <td>0.567824</td>
      <td>5.176471</td>
      <td>-9.349529</td>
      <td>0.647059</td>
      <td>0.362829</td>
      <td>0.336008</td>
      <td>...</td>
      <td>126.193529</td>
      <td>3.882353</td>
      <td>184112.7059</td>
      <td>3129916</td>
      <td>17</td>
      <td>world</td>
      <td>43</td>
      <td>5.352941</td>
      <td>3.068551</td>
      <td>52.165371</td>
    </tr>
    <tr>
      <th>1576</th>
      <td>Intervju med Linnea Henriksson</td>
      <td>37i9dQZF1DXd0KoQoKcF6I</td>
      <td>70</td>
      <td>0.644111</td>
      <td>0.605778</td>
      <td>5.833333</td>
      <td>-8.970389</td>
      <td>0.666667</td>
      <td>0.345078</td>
      <td>0.429661</td>
      <td>...</td>
      <td>112.738444</td>
      <td>4.000000</td>
      <td>186260.0000</td>
      <td>3352680</td>
      <td>18</td>
      <td>indie</td>
      <td>43</td>
      <td>19.222222</td>
      <td>3.104340</td>
      <td>55.878112</td>
    </tr>
    <tr>
      <th>1577</th>
      <td>Intervju med Anders FridÌ©n</td>
      <td>37i9dQZF1DX16NkJCLGI9I</td>
      <td>75</td>
      <td>0.547750</td>
      <td>0.707812</td>
      <td>5.562500</td>
      <td>-8.736500</td>
      <td>0.500000</td>
      <td>0.336213</td>
      <td>0.310332</td>
      <td>...</td>
      <td>118.959625</td>
      <td>3.750000</td>
      <td>215204.4375</td>
      <td>3443271</td>
      <td>16</td>
      <td>metal</td>
      <td>47</td>
      <td>4.562500</td>
      <td>3.586748</td>
      <td>57.387965</td>
    </tr>
    <tr>
      <th>1578</th>
      <td>Intervju med Kristian Anttila</td>
      <td>37i9dQZF1DWTM7OuHqfCXB</td>
      <td>40</td>
      <td>0.534722</td>
      <td>0.526667</td>
      <td>7.000000</td>
      <td>-9.908056</td>
      <td>0.666667</td>
      <td>0.327894</td>
      <td>0.430030</td>
      <td>...</td>
      <td>120.497833</td>
      <td>3.944444</td>
      <td>172444.8333</td>
      <td>3104007</td>
      <td>18</td>
      <td>indie</td>
      <td>0</td>
      <td>0.000000</td>
      <td>2.874086</td>
      <td>51.733553</td>
    </tr>
    <tr>
      <th>1579</th>
      <td>Intervju med Veronica Maggio</td>
      <td>37i9dQZF1DX0GyWkLSMHBw</td>
      <td>138</td>
      <td>0.576063</td>
      <td>0.521062</td>
      <td>5.437500</td>
      <td>-8.976562</td>
      <td>0.625000</td>
      <td>0.257288</td>
      <td>0.474981</td>
      <td>...</td>
      <td>128.648188</td>
      <td>3.875000</td>
      <td>178993.9375</td>
      <td>2863903</td>
      <td>16</td>
      <td>pop</td>
      <td>56</td>
      <td>31.562500</td>
      <td>2.983238</td>
      <td>47.731812</td>
    </tr>
    <tr>
      <th>1580</th>
      <td>Hemma Hos Janne&amp;Kjell</td>
      <td>37i9dQZF1DWUQDxrwmYosp</td>
      <td>253</td>
      <td>0.632469</td>
      <td>0.315196</td>
      <td>5.571429</td>
      <td>-16.543082</td>
      <td>0.673469</td>
      <td>0.852245</td>
      <td>0.753098</td>
      <td>...</td>
      <td>95.441327</td>
      <td>3.367347</td>
      <td>281788.5918</td>
      <td>13807641</td>
      <td>49</td>
      <td>None</td>
      <td>6</td>
      <td>1.653061</td>
      <td>4.696486</td>
      <td>230.127810</td>
    </tr>
    <tr>
      <th>1581</th>
      <td>Skratt till kaffet</td>
      <td>37i9dQZF1DWXiBKWKeM4cZ</td>
      <td>471</td>
      <td>0.602542</td>
      <td>0.579000</td>
      <td>6.125000</td>
      <td>-14.646833</td>
      <td>0.583333</td>
      <td>0.894042</td>
      <td>0.832875</td>
      <td>...</td>
      <td>100.532583</td>
      <td>3.583333</td>
      <td>252897.2083</td>
      <td>6069533</td>
      <td>24</td>
      <td>None</td>
      <td>29</td>
      <td>19.750000</td>
      <td>4.214962</td>
      <td>101.159086</td>
    </tr>
    <tr>
      <th>1582</th>
      <td>International Playboy</td>
      <td>73boXMJz9iBoXxQVFZ94r5</td>
      <td>21906</td>
      <td>0.613783</td>
      <td>0.738957</td>
      <td>6.304348</td>
      <td>-6.556304</td>
      <td>0.478261</td>
      <td>0.223004</td>
      <td>0.140342</td>
      <td>...</td>
      <td>125.770739</td>
      <td>4.000000</td>
      <td>269375.4783</td>
      <td>6195636</td>
      <td>23</td>
      <td>pop</td>
      <td>71</td>
      <td>38.434783</td>
      <td>4.489600</td>
      <td>103.260807</td>
    </tr>
    <tr>
      <th>1583</th>
      <td>National Blood Week</td>
      <td>2y74Ha0ztRUcIYmqITh0D4</td>
      <td>8</td>
      <td>0.527560</td>
      <td>0.702880</td>
      <td>4.880000</td>
      <td>-7.700760</td>
      <td>0.600000</td>
      <td>0.049536</td>
      <td>0.229692</td>
      <td>...</td>
      <td>125.620600</td>
      <td>3.960000</td>
      <td>246554.4400</td>
      <td>6163861</td>
      <td>25</td>
      <td>rock</td>
      <td>76</td>
      <td>37.800000</td>
      <td>4.109249</td>
      <td>102.731222</td>
    </tr>
    <tr>
      <th>1584</th>
      <td>Most Listened To British Dads on Spotify</td>
      <td>1k9jG0FUp7BcrAF1MZSabO</td>
      <td>5</td>
      <td>0.533350</td>
      <td>0.723300</td>
      <td>5.650000</td>
      <td>-7.356650</td>
      <td>0.900000</td>
      <td>0.052920</td>
      <td>0.108718</td>
      <td>...</td>
      <td>128.547450</td>
      <td>3.850000</td>
      <td>265870.2000</td>
      <td>5317404</td>
      <td>20</td>
      <td>rock</td>
      <td>78</td>
      <td>57.200000</td>
      <td>4.431179</td>
      <td>88.623577</td>
    </tr>
    <tr>
      <th>1585</th>
      <td>Viral Hits</td>
      <td>37i9dQZF1DX44t7uCdkV1A</td>
      <td>526827</td>
      <td>0.703225</td>
      <td>0.579100</td>
      <td>5.875000</td>
      <td>-6.979500</td>
      <td>0.675000</td>
      <td>0.110942</td>
      <td>0.245210</td>
      <td>...</td>
      <td>124.394500</td>
      <td>3.850000</td>
      <td>215071.4750</td>
      <td>8602859</td>
      <td>40</td>
      <td>None</td>
      <td>91</td>
      <td>70.400000</td>
      <td>3.584532</td>
      <td>143.381270</td>
    </tr>
    <tr>
      <th>1586</th>
      <td>Essential Folk</td>
      <td>37i9dQZF1DWVmps5U8gHNv</td>
      <td>250576</td>
      <td>0.516149</td>
      <td>0.370010</td>
      <td>4.797872</td>
      <td>-13.188372</td>
      <td>0.861702</td>
      <td>0.045100</td>
      <td>0.613671</td>
      <td>...</td>
      <td>117.459106</td>
      <td>3.829787</td>
      <td>225510.1809</td>
      <td>21197957</td>
      <td>94</td>
      <td>indie</td>
      <td>73</td>
      <td>43.648936</td>
      <td>3.758511</td>
      <td>353.299990</td>
    </tr>
    <tr>
      <th>1587</th>
      <td>Women of Pop</td>
      <td>37i9dQZF1DX3WvGXE8FqYX</td>
      <td>401054</td>
      <td>0.500389</td>
      <td>0.536944</td>
      <td>3.769841</td>
      <td>-5.037238</td>
      <td>0.492063</td>
      <td>0.056767</td>
      <td>0.148345</td>
      <td>...</td>
      <td>94.271325</td>
      <td>3.174603</td>
      <td>181182.6825</td>
      <td>22829018</td>
      <td>126</td>
      <td>pop</td>
      <td>79</td>
      <td>37.238095</td>
      <td>3.019717</td>
      <td>380.484394</td>
    </tr>
    <tr>
      <th>1588</th>
      <td>dw-c</td>
      <td>5ji4GZJpll6twskFvKxiHx</td>
      <td>4</td>
      <td>0.669720</td>
      <td>0.618930</td>
      <td>5.680000</td>
      <td>-6.423740</td>
      <td>0.520000</td>
      <td>0.097666</td>
      <td>0.206085</td>
      <td>...</td>
      <td>114.548960</td>
      <td>4.060000</td>
      <td>217227.8000</td>
      <td>10861390</td>
      <td>50</td>
      <td>pop</td>
      <td>82</td>
      <td>60.240000</td>
      <td>3.620471</td>
      <td>181.023529</td>
    </tr>
    <tr>
      <th>1589</th>
      <td>dw_g</td>
      <td>40VxbK9NqccdUDUpiUXmbp</td>
      <td>2</td>
      <td>0.597000</td>
      <td>0.523767</td>
      <td>5.166667</td>
      <td>-8.538900</td>
      <td>0.633333</td>
      <td>0.075450</td>
      <td>0.381130</td>
      <td>...</td>
      <td>112.383500</td>
      <td>3.866667</td>
      <td>229589.3333</td>
      <td>6887680</td>
      <td>30</td>
      <td>pop</td>
      <td>71</td>
      <td>53.933333</td>
      <td>3.826497</td>
      <td>114.794896</td>
    </tr>
    <tr>
      <th>1590</th>
      <td>Top Shower Songs</td>
      <td>0RTz1jFo5BXGPfI8eVf8sj</td>
      <td>21</td>
      <td>0.672650</td>
      <td>0.729610</td>
      <td>5.640000</td>
      <td>-6.017110</td>
      <td>0.730000</td>
      <td>0.087193</td>
      <td>0.142181</td>
      <td>...</td>
      <td>121.881970</td>
      <td>4.000000</td>
      <td>225373.6300</td>
      <td>22537363</td>
      <td>100</td>
      <td>pop</td>
      <td>81</td>
      <td>66.340000</td>
      <td>3.756235</td>
      <td>375.623468</td>
    </tr>
    <tr>
      <th>1591</th>
      <td>foodora dinner playlist</td>
      <td>4lgseztVwmKQ8MNETPVIny</td>
      <td>22</td>
      <td>0.597560</td>
      <td>0.373000</td>
      <td>5.240000</td>
      <td>-9.471560</td>
      <td>0.760000</td>
      <td>0.057216</td>
      <td>0.606060</td>
      <td>...</td>
      <td>115.777000</td>
      <td>3.720000</td>
      <td>238859.1200</td>
      <td>5971478</td>
      <td>25</td>
      <td>pop</td>
      <td>85</td>
      <td>70.920000</td>
      <td>3.980993</td>
      <td>99.524832</td>
    </tr>
  </tbody>
</table>
<p>1592 rows × 23 columns</p>
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





    <matplotlib.text.Text at 0x120ffacc0>




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





    <matplotlib.text.Text at 0x121446438>




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

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.scatter(df["total_num"], df["followers"])
ax1.set_ylabel('Followers')
ax1.set_xlabel('Number of Songs')
```





    <matplotlib.text.Text at 0x121bbc3c8>




![png](Final%2BProject%2BEDA_files/Final%2BProject%2BEDA_12_1.png)



![png](Final%2BProject%2BEDA_files/Final%2BProject%2BEDA_12_2.png)


#### A quick test for colinearity provides us with good information about the information that our predictors encode not just individually, but as an overall dataset. We wanted to look at these relationships as we will need to be aware of what predictors are the most powerful and which can be encapsulated by another variable.



```python
from pandas.tools.plotting import scatter_matrix
fig, ax = plt.subplots(1,1, figsize=(15,10))
axes = scatter_matrix(df[["average_danceability","average_energy","average_loudness","average_valence",
                          "average_tempo","average_time_signature"]], alpha=0.5, diagonal='kde', ax=ax)
plt.show()
```


    /anaconda/lib/python3.6/site-packages/ipykernel_launcher.py:4: FutureWarning: 'pandas.tools.plotting.scatter_matrix' is deprecated, import 'pandas.plotting.scatter_matrix' instead.
      after removing the cwd from sys.path.
    /anaconda/lib/python3.6/site-packages/ipykernel_launcher.py:4: UserWarning: To output multiple subplots, the figure containing the passed axes is being cleared
      after removing the cwd from sys.path.



![png](Final%2BProject%2BEDA_files/Final%2BProject%2BEDA_14_1.png)


#### Since we will be trying to create an optimal playlist queried by genre, we decided to catgorize our playlists into 23 genres that described the artists on the tracks included in the playlist. Spotify API did not assign genres to actual playlists so we imputed them through the artist and generalized so that we would have a significant amount of playlists in each genre.



```python
genre_info = ['average_danceability','average_energy','average_key','average_loudness',
              'average_mode','average_speechiness','average_acousticness','average_instrumentalness',
              'average_liveness','average_valence','average_tempo','average_time_signature',
              'average_duration_ms','total_duration','max_popularity','avg_popularity','average_duration_min',
              'total_duration_min']
df_cats_minus_audio = df.drop(genre_info, axis=1)
df_cats_minus_audio
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
    <tr>
      <th>5</th>
      <td>Hot Country</td>
      <td>37i9dQZF1DX1lVhptIYRda</td>
      <td>4244887</td>
      <td>51</td>
      <td>country</td>
    </tr>
    <tr>
      <th>6</th>
      <td>åÁViva Latino!</td>
      <td>37i9dQZF1DX10zKzsJ2jva</td>
      <td>6607264</td>
      <td>50</td>
      <td>world</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Afternoon Acoustic</td>
      <td>37i9dQZF1DX4E3UdUs7fUx</td>
      <td>2470127</td>
      <td>81</td>
      <td>indie</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Peaceful Piano</td>
      <td>37i9dQZF1DX4sWSpwq3LiO</td>
      <td>3313734</td>
      <td>162</td>
      <td>ambient</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Roots Rising</td>
      <td>37i9dQZF1DWYV7OOaGhoH0</td>
      <td>987430</td>
      <td>105</td>
      <td>indie</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Lost In Austin: Country Music from SXSW</td>
      <td>5JdIoXkjfuyWmDdDnznmq1</td>
      <td>3822</td>
      <td>82</td>
      <td>None</td>
    </tr>
    <tr>
      <th>11</th>
      <td>SXSW: HipHopUpNext</td>
      <td>37i9dQZF1DX5PuwKY2VZBZ</td>
      <td>14524</td>
      <td>25</td>
      <td>rap</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Women of SXSW</td>
      <td>37i9dQZF1DXboGLA1CNczK</td>
      <td>3352</td>
      <td>55</td>
      <td>None</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Music Inspired By Mogul</td>
      <td>4HghCZXykAY0kdRqJipQJd</td>
      <td>3638</td>
      <td>21</td>
      <td>rap</td>
    </tr>
    <tr>
      <th>14</th>
      <td>SXSW 2017: Daily Guide</td>
      <td>1LYK2ahDnbLnS4fddCFY7z</td>
      <td>7879</td>
      <td>55</td>
      <td>rock</td>
    </tr>
    <tr>
      <th>15</th>
      <td>TGIF</td>
      <td>37i9dQZF1DXcRXFNfZr7Tp</td>
      <td>1454537</td>
      <td>50</td>
      <td>electronic</td>
    </tr>
    <tr>
      <th>16</th>
      <td>SXSW UK Rising</td>
      <td>37i9dQZF1DXdIcOzOLoAn6</td>
      <td>8186</td>
      <td>51</td>
      <td>None</td>
    </tr>
    <tr>
      <th>17</th>
      <td>ExperiWOMENtell</td>
      <td>37i9dQZF1DWY7uXxMycXfq</td>
      <td>1726</td>
      <td>40</td>
      <td>other</td>
    </tr>
    <tr>
      <th>18</th>
      <td>The Refugee Playlist</td>
      <td>37i9dQZF1DX1328t2iygZy</td>
      <td>16047</td>
      <td>22</td>
      <td>pop</td>
    </tr>
    <tr>
      <th>19</th>
      <td>This Is: Jason Isbell</td>
      <td>37i9dQZF1DX1wvY0jqbS55</td>
      <td>11349</td>
      <td>49</td>
      <td>country</td>
    </tr>
    <tr>
      <th>20</th>
      <td>This Is: Disney</td>
      <td>37i9dQZF1DWVs8I62NcHks</td>
      <td>247039</td>
      <td>58</td>
      <td>soundtrack</td>
    </tr>
    <tr>
      <th>21</th>
      <td>This Is: The Doors</td>
      <td>37i9dQZF1DX95gx8SY6DLX</td>
      <td>82684</td>
      <td>50</td>
      <td>rock</td>
    </tr>
    <tr>
      <th>22</th>
      <td>This Is: AC/DC</td>
      <td>37i9dQZF1DXec50AjHrNTq</td>
      <td>522221</td>
      <td>35</td>
      <td>rock</td>
    </tr>
    <tr>
      <th>23</th>
      <td>This Is: Adele</td>
      <td>37i9dQZF1DWZUozJiHy44Y</td>
      <td>527006</td>
      <td>25</td>
      <td>pop</td>
    </tr>
    <tr>
      <th>24</th>
      <td>This Is: Al Green</td>
      <td>37i9dQZF1DWZMCPjHG57gq</td>
      <td>27122</td>
      <td>26</td>
      <td>rock</td>
    </tr>
    <tr>
      <th>25</th>
      <td>This Is: alt-J</td>
      <td>37i9dQZF1DWXMmak2OV7PN</td>
      <td>62309</td>
      <td>23</td>
      <td>rock</td>
    </tr>
    <tr>
      <th>26</th>
      <td>This Is: Aqua</td>
      <td>37i9dQZF1DX3reMgdkcDa4</td>
      <td>7346</td>
      <td>21</td>
      <td>dance</td>
    </tr>
    <tr>
      <th>27</th>
      <td>This is: Aretha Franklin</td>
      <td>37i9dQZF1DX6bJVMtDYJHx</td>
      <td>26987</td>
      <td>50</td>
      <td>oldies</td>
    </tr>
    <tr>
      <th>28</th>
      <td>This Is: Bee Gees</td>
      <td>37i9dQZF1DWZvGllBOMzci</td>
      <td>118382</td>
      <td>52</td>
      <td>rock</td>
    </tr>
    <tr>
      <th>29</th>
      <td>This Is: blink-182</td>
      <td>37i9dQZF1DX4EKARgqYPFZ</td>
      <td>130409</td>
      <td>65</td>
      <td>christmas</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1562</th>
      <td>88 Keys</td>
      <td>37i9dQZF1DX561TxkFttR4</td>
      <td>220709</td>
      <td>88</td>
      <td>ambient</td>
    </tr>
    <tr>
      <th>1563</th>
      <td>Raw Expression</td>
      <td>3yUO32lGCEnAfFGORSEsy4</td>
      <td>0</td>
      <td>42</td>
      <td>dance</td>
    </tr>
    <tr>
      <th>1564</th>
      <td>Sunshine Reggae</td>
      <td>37i9dQZF1DXbwoaqxaoAVr</td>
      <td>159270</td>
      <td>50</td>
      <td>pop</td>
    </tr>
    <tr>
      <th>1565</th>
      <td>Morning Rhythm</td>
      <td>37i9dQZF1DX3ohNxI5tB79</td>
      <td>115660</td>
      <td>50</td>
      <td>indie</td>
    </tr>
    <tr>
      <th>1566</th>
      <td>I Love My Neo Soul</td>
      <td>37i9dQZF1DX44dZ4p5QLf4</td>
      <td>96655</td>
      <td>49</td>
      <td>soul</td>
    </tr>
    <tr>
      <th>1567</th>
      <td>Afternoon Train Ride</td>
      <td>37i9dQZF1DX8MbMfAHb8U0</td>
      <td>25081</td>
      <td>65</td>
      <td>soul</td>
    </tr>
    <tr>
      <th>1568</th>
      <td>The Midnight Hour</td>
      <td>37i9dQZF1DXea80XwOJRgD</td>
      <td>50982</td>
      <td>46</td>
      <td>blues</td>
    </tr>
    <tr>
      <th>1569</th>
      <td>Soultronic</td>
      <td>37i9dQZF1DXdc0DUqaW3MZ</td>
      <td>99996</td>
      <td>50</td>
      <td>None</td>
    </tr>
    <tr>
      <th>1570</th>
      <td>The Cookout</td>
      <td>37i9dQZF1DXab8DipvnuNU</td>
      <td>141749</td>
      <td>47</td>
      <td>blues</td>
    </tr>
    <tr>
      <th>1571</th>
      <td>Icona Pops guide till festivalpackningen</td>
      <td>37i9dQZF1DXdwuuZvpX2qk</td>
      <td>21</td>
      <td>24</td>
      <td>None</td>
    </tr>
    <tr>
      <th>1572</th>
      <td>MidsommarvÌ_rd: Tomas Ledin</td>
      <td>37i9dQZF1DX4uFp6ryXvMm</td>
      <td>638</td>
      <td>30</td>
      <td>None</td>
    </tr>
    <tr>
      <th>1573</th>
      <td>Intervju med Seinabo Sey</td>
      <td>2k9EArKbf7N3QUmuNJHSo8</td>
      <td>2</td>
      <td>3</td>
      <td>pop</td>
    </tr>
    <tr>
      <th>1574</th>
      <td>Intervju med Mando Diao</td>
      <td>37i9dQZF1DWZHJcG9TkzIK</td>
      <td>81</td>
      <td>15</td>
      <td>rock</td>
    </tr>
    <tr>
      <th>1575</th>
      <td>Intervju med Agnes</td>
      <td>37i9dQZF1DX8uJwKSyPfRr</td>
      <td>104</td>
      <td>17</td>
      <td>world</td>
    </tr>
    <tr>
      <th>1576</th>
      <td>Intervju med Linnea Henriksson</td>
      <td>37i9dQZF1DXd0KoQoKcF6I</td>
      <td>70</td>
      <td>18</td>
      <td>indie</td>
    </tr>
    <tr>
      <th>1577</th>
      <td>Intervju med Anders FridÌ©n</td>
      <td>37i9dQZF1DX16NkJCLGI9I</td>
      <td>75</td>
      <td>16</td>
      <td>metal</td>
    </tr>
    <tr>
      <th>1578</th>
      <td>Intervju med Kristian Anttila</td>
      <td>37i9dQZF1DWTM7OuHqfCXB</td>
      <td>40</td>
      <td>18</td>
      <td>indie</td>
    </tr>
    <tr>
      <th>1579</th>
      <td>Intervju med Veronica Maggio</td>
      <td>37i9dQZF1DX0GyWkLSMHBw</td>
      <td>138</td>
      <td>16</td>
      <td>pop</td>
    </tr>
    <tr>
      <th>1580</th>
      <td>Hemma Hos Janne&amp;Kjell</td>
      <td>37i9dQZF1DWUQDxrwmYosp</td>
      <td>253</td>
      <td>49</td>
      <td>None</td>
    </tr>
    <tr>
      <th>1581</th>
      <td>Skratt till kaffet</td>
      <td>37i9dQZF1DWXiBKWKeM4cZ</td>
      <td>471</td>
      <td>24</td>
      <td>None</td>
    </tr>
    <tr>
      <th>1582</th>
      <td>International Playboy</td>
      <td>73boXMJz9iBoXxQVFZ94r5</td>
      <td>21906</td>
      <td>23</td>
      <td>pop</td>
    </tr>
    <tr>
      <th>1583</th>
      <td>National Blood Week</td>
      <td>2y74Ha0ztRUcIYmqITh0D4</td>
      <td>8</td>
      <td>25</td>
      <td>rock</td>
    </tr>
    <tr>
      <th>1584</th>
      <td>Most Listened To British Dads on Spotify</td>
      <td>1k9jG0FUp7BcrAF1MZSabO</td>
      <td>5</td>
      <td>20</td>
      <td>rock</td>
    </tr>
    <tr>
      <th>1585</th>
      <td>Viral Hits</td>
      <td>37i9dQZF1DX44t7uCdkV1A</td>
      <td>526827</td>
      <td>40</td>
      <td>None</td>
    </tr>
    <tr>
      <th>1586</th>
      <td>Essential Folk</td>
      <td>37i9dQZF1DWVmps5U8gHNv</td>
      <td>250576</td>
      <td>94</td>
      <td>indie</td>
    </tr>
    <tr>
      <th>1587</th>
      <td>Women of Pop</td>
      <td>37i9dQZF1DX3WvGXE8FqYX</td>
      <td>401054</td>
      <td>126</td>
      <td>pop</td>
    </tr>
    <tr>
      <th>1588</th>
      <td>dw-c</td>
      <td>5ji4GZJpll6twskFvKxiHx</td>
      <td>4</td>
      <td>50</td>
      <td>pop</td>
    </tr>
    <tr>
      <th>1589</th>
      <td>dw_g</td>
      <td>40VxbK9NqccdUDUpiUXmbp</td>
      <td>2</td>
      <td>30</td>
      <td>pop</td>
    </tr>
    <tr>
      <th>1590</th>
      <td>Top Shower Songs</td>
      <td>0RTz1jFo5BXGPfI8eVf8sj</td>
      <td>21</td>
      <td>100</td>
      <td>pop</td>
    </tr>
    <tr>
      <th>1591</th>
      <td>foodora dinner playlist</td>
      <td>4lgseztVwmKQ8MNETPVIny</td>
      <td>22</td>
      <td>25</td>
      <td>pop</td>
    </tr>
  </tbody>
</table>
<p>1592 rows × 5 columns</p>
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
