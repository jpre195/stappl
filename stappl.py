import streamlit as st
import pandas as pd
import scipy
import numpy as np

st.title('Statistical Testing APPLication')

tab1, tab2 = st.tabs([':house: Home', 'Z-Test'])

#########
# Tab 1 #
#########

tab1.write('Hello World')

file = tab1.file_uploader('Choose a file', type = ['csv'])

df = pd.DataFrame() if file is None else pd.read_csv(file)

# if file is not None:

#     df = pd.read_csv(file)

tab1.dataframe(df)

#########
# Tab 2 #
#########

sided_test = tab2.selectbox('Test Type', ['Left-sided', 'Right-sided', '2-sided'])
mu = tab2.number_input('Population mean')
sigma = tab2.number_input('Population standard deviation', value = 1)
alpha = tab2.number_input('Significance level', value = 0.05)

sample_mean = df.mean()

z_score = (sample_mean - mu) / (sigma / np.sqrt(df.shape[0]))

if sided_test == 'Left-sided':

    p_value = scipy.stats.norm.cdf(z_score)

elif sided_test == 'Right-sided':

    p_value = 1 - scipy.stats.norm.cdf(z_score)

else:

    p_value = 2 * (1 - scipy.stats.norm.cdf(abs(z_score)))

if p_value < alpha:

    result = f'Since p-value = {round(float(p_value), 3)} < {alpha}, we reject the null'

else:

    result = f'Since p-value = {round(float(p_value), 3)} >= {alpha}, we fail to reject the null'

tab2.write(result)