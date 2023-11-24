import streamlit as st
import pandas as pd
import scipy
import numpy as np
import plotly.graph_objects as go

#Wide mode
st.set_page_config(layout = 'wide')

#Title of page
st.title('Z-Test')

#Read in dataframe
df = pd.DataFrame() if 'df' not in st.session_state else st.session_state['df']

#Expander of test parameters
with st.expander('Test parameters', expanded = True):

    sided_test = st.selectbox('Test Type', ['Left-sided', 'Right-sided', '2-sided'])
    mu = st.number_input('Population mean ($\mu_0$)')
    sigma = st.number_input('Population standard deviation ($\sigma$)', value = 1)
    alpha = st.number_input('Significance level ($\\alpha$)', value = 0.05)
    sample = st.selectbox('Data', df.columns)

#Alternative hypotheses
h1s = {'Left-sided' : f'$H_1: \mu < {mu}$',
       'Right-sided' : f'$H_1: \mu > {mu}$',
       '2-sided' : f'$H_1: \mu \\neq {mu}$'}

st.divider()

col1, col2 = st.columns(2)

#Write null and alternative hypothesis
col1.header('Hypotheses')
col1.write(f'$$H_0: \mu = {mu}$$')
col1.write(h1s[sided_test])

#Calculate mean
sample_mean = df[sample].mean()

#Write samples mean
col2.header('Sample Statistics')
col2.write('$\overline{X} = ' + str(round(sample_mean, 4)) + '$')

st.divider()

#Calculate z-score
z_score = (sample_mean - mu) / (sigma / np.sqrt(df.shape[0]))
# z_score = list(z_score)[0]

#Calculate p-value
if sided_test == 'Left-sided':

    p_value = scipy.stats.norm.cdf(z_score)

elif sided_test == 'Right-sided':

    p_value = 1 - scipy.stats.norm.cdf(z_score)

else:

    p_value = 2 * (1 - scipy.stats.norm.cdf(abs(z_score)))

#Interpret results
if p_value < alpha:

    result = f'Since p-value $ = {round(float(p_value), 4)} < \\alpha = {alpha}$, we __reject the null hypothesis__ in favor of the alternative (${h1s[sided_test].split(":")[-1]})'

else:

    result = f'Since p-value $ = {round(float(p_value), 4)} \geq \\alpha = {alpha}$, we __fail to reject the null hypothesis__ that $\mu = {mu}$'

col1, col2 = st.columns([1, 2])

#Write conclusion
col1.header('Conclusion')
col1.write(result)

#Create x and y-axis for plot
x_axis = np.arange(-3, 3, step = 6 / 1000)
y_axis = scipy.stats.norm.pdf(x_axis)

#Plot normal PDF
line_trace = go.Line(x = x_axis, y = y_axis, fillcolor = 'blue')

if sided_test == '2-sided':

    #Area under z-score
    highlight_x1 = np.arange(np.abs(z_score), 3, (3 - np.abs(z_score)) / 1000)
    highlight_x2 = np.arange(-3, -1 * np.abs(z_score), (3 - np.abs(z_score)) / 1000)
    highlight_y1 = scipy.stats.norm.pdf(highlight_x1)
    highlight_y2 = scipy.stats.norm.pdf(highlight_x2)

    #Critical alphas
    z_alphas = scipy.stats.norm.ppf([alpha / 2, 1 - (alpha / 2)])

    #Area traces
    area_trace1 = go.Line(x = highlight_x1, y = highlight_y1, fill = 'tozeroy', fillcolor = 'blue')
    area_trace2 = go.Line(x = highlight_x2, y = highlight_y2, fill = 'tozeroy', fillcolor = 'blue')

    #Create figure
    fig = go.Figure(data = [line_trace, area_trace1, area_trace2])

    #Add vertical lines for z-score and critical value
    fig.add_vline(-1 * z_score, line_dash = 'dash', line_color = 'blue')
    fig.add_vline(z_score, line_dash = 'dash', line_color = 'blue')
    fig.add_vline(z_alphas[0], line_dash = 'dash', line_color = 'red')
    fig.add_vline(z_alphas[1], line_dash = 'dash', line_color = 'red')

    #Hide legend and add tickmarks
    fig.update_layout(showlegend = False)
    fig.update_xaxes(tickvals = [i for i in range(-3, 4)] + [z_score] + list(z_alphas),
                    ticktext = [i for i in range(-3, 4)] + ['Z-Score'] + ['Z-Critical', 'Z-Critical'])

else:

    if sided_test == 'Left-sided':

        #Values left of z-score
        highlight_x = np.arange(-3, z_score, (z_score + 3) / 1000)
        z_alpha = scipy.stats.norm.ppf(alpha)

    elif sided_test == 'Right-sided':
    
        #Values right of z-score
        highlight_x = np.arange(z_score, 3, (3 - z_score) / 1000)
        z_alpha = scipy.stats.norm.ppf(1 - alpha)

    #PDF of x-values
    highlight_y = scipy.stats.norm.pdf(highlight_x)

    #Area under z-score
    area_trace = go.Line(x = highlight_x, y = highlight_y, fill = 'tozeroy', fillcolor = 'blue')

    #Create figure
    fig = go.Figure(data = [line_trace, area_trace])

    #Vertical lines for z-score and critical value
    fig.add_vline(z_score, line_dash = 'dash', line_color = 'blue')
    fig.add_vline(z_alpha, line_dash = 'dash', line_color = 'red')

    #Hide legend and add tick marks
    fig.update_layout(showlegend = False)
    fig.update_xaxes(tickvals = [i for i in range(-3, 4)] + [z_score, z_alpha],
                    ticktext = [i for i in range(-3, 4)] + ['Z-Score', 'Z-Critical'])

#Show plot
col2.plotly_chart(fig, use_container_width = True)