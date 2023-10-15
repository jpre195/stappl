import streamlit as st
import pandas as pd
import scipy
import numpy as np
import matplotlib.pyplot as plt
import altair as alt

st.title('Statistical Testing APPLication')

file = st.file_uploader('Choose a file', type = ['csv'])

with st.expander('Test parameters', expanded = True):

    sided_test = st.selectbox('Test Type', ['Left-sided', 'Right-sided', '2-sided'])
    # sided_test = st.selectbox('Alternative Hypothesis', ['$\mu < 0$', '$\mu > 0$', '$\mu \neq 0$'])
    mu = st.number_input('Population mean ($\mu_0$)')
    sigma = st.number_input('Population standard deviation ($\sigma$)', value = 1)
    alpha = st.number_input('Significance level ($\\alpha$)', value = 0.05)

h1s = {'Left-sided' : f'$H_1: \mu < {mu}$',
       'Right-sided' : f'$H_1: \mu > {mu}$',
       '2-sided' : f'$H_1: \mu \\neq {mu}$'}

st.divider()
st.header('Hypotheses')
st.write(f'$$H_0: \mu = {mu}$$')
st.write(h1s[sided_test])
st.divider()

if file is not None:

    df = pd.DataFrame() if file is None else pd.read_csv(file)

    # st.dataframe(df)

    sample_mean = df.mean()

    st.header('Sample Statistics')
    st.write('$\overline{X} = ' + str(round(sample_mean.values[0], 4)) + '$')
    st.divider()

    z_score = (sample_mean - mu) / (sigma / np.sqrt(df.shape[0]))

    if sided_test == 'Left-sided':

        p_value = scipy.stats.norm.cdf(z_score)

    elif sided_test == 'Right-sided':

        p_value = 1 - scipy.stats.norm.cdf(z_score)

    else:

        p_value = 2 * (1 - scipy.stats.norm.cdf(abs(z_score)))

    if p_value < alpha:

        result = f'Since p-value $ = {round(float(p_value), 4)} < \\alpha = {alpha}$, we __reject the null hypothesis__ in favor of the alternative (${h1s[sided_test].split(":")[-1]})'

    else:

        result = f'Since p-value $ = {round(float(p_value), 4)} \geq \\alpha = {alpha}$, we __fail to reject the null hypothesis__ that $\mu = {mu}$'

    st.header('Conclusion')
    st.write(result)

    x_axis = np.arange(-3, 3, step = 6 / 1000)

    plot_df = pd.DataFrame({'x' : x_axis,
                            'pdf' : scipy.stats.norm.pdf(x_axis)})

    chart = (alt.Chart(plot_df)
                    .mark_line()
                    .encode(
                        x = 'x',
                        y = 'pdf'
                    )
            )
    
    points_df = pd.DataFrame({'x' : [alpha, sample_mean]})
    
    sample_mean_chart = (alt.Chart(points_df)
                         .mark_rule()
                         .encode(x = 'x'))

    # st.altair_chart(chart + sample_mean_chart, use_container_width = True)
