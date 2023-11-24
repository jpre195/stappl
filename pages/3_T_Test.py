import streamlit as st
import pandas as pd
import scipy
import numpy as np
import plotly.graph_objects as go

#Wide mode
st.set_page_config(layout = 'wide')

#Title of page
st.title('T-Test')

#Read in dataframe
df = pd.DataFrame() if 'df' not in st.session_state else st.session_state['df']

tab1, tab2, tab3 = st.tabs(['One-Sample', 'Paired', 'Independent Samples'])

###########
## Tab 1 ##
###########

#Expander of test parameters
with tab1.expander('Test parameters', expanded = True):

    sided_test = st.selectbox('Test Type', ['Left-sided', 'Right-sided', '2-sided'], key = 'one_sample_select')
    mu = st.number_input('Population mean ($\mu_0$)')
    alpha = st.number_input('Significance level ($\\alpha$)', value = 0.05, key = 'one_sample_alpha')
    sample = st.selectbox('Data', df.columns)

#Alternative hypotheses
h1s = {'Left-sided' : f'$H_1: \mu < {mu}$',
       'Right-sided' : f'$H_1: \mu > {mu}$',
       '2-sided' : f'$H_1: \mu \\neq {mu}$'}

tab1.divider()

col1, col2, col3 = tab1.columns(3)

#Write null and alternative hypothesis
col1.header('Hypotheses')
col1.write(f'$$H_0: \mu = {mu}$$')
col1.write(h1s[sided_test])

#Calculate mean
sample_mean = df[sample].mean()
sample_std = df[sample].std()

#Write samples mean
col2.header('Sample Statistics')
col2.write('$\overline{X} = ' + str(round(sample_mean, 4)) + '$')
col2.write('$S^2 = ' + str(round(sample_std, 4)) + '$')

#Calculate z-score
t_score = (sample_mean - mu) / (sample_std / np.sqrt(df.shape[0]))

col3.header('Test Statistic')
col3.write('$T = ' + str(round(t_score, 4)) + '$')

tab1.divider()

#Calculate p-value
if sided_test == 'Left-sided':

    p_value = scipy.stats.t.cdf(t_score, df.shape[0] - 1)

elif sided_test == 'Right-sided':

    p_value = 1 - scipy.stats.t.cdf(t_score, df.shape[0] - 1)

else:

    p_value = 2 * (1 - scipy.stats.t.cdf(abs(t_score), df.shape[0] - 1))

#Interpret results
if p_value < alpha:

    result = f'Since p-value $ = {round(float(p_value), 4)} < \\alpha = {alpha}$, we __reject the null hypothesis__ in favor of the alternative (${h1s[sided_test].split(":")[-1]})'

else:

    result = f'Since p-value $ = {round(float(p_value), 4)} \geq \\alpha = {alpha}$, we __fail to reject the null hypothesis__ that $\mu = {mu}$'

col1, col2 = tab1.columns([1, 2])

#Write conclusion
col1.header('Conclusion')
col1.write(result)

#Create x and y-axis for plot
x_axis = np.arange(-3, 3, step = 6 / 1000)
y_axis = scipy.stats.t.pdf(x_axis, df.shape[0] - 1)

#Plot normal PDF
line_trace = go.Line(x = x_axis, y = y_axis, fillcolor = 'blue')

if sided_test == '2-sided':

    #Area under z-score
    highlight_x1 = np.arange(np.abs(t_score), 3, (3 - np.abs(t_score)) / 1000)
    highlight_x2 = np.arange(-3, -1 * np.abs(t_score), (3 - np.abs(t_score)) / 1000)
    highlight_y1 = scipy.stats.t.pdf(highlight_x1, df.shape[0] - 1)
    highlight_y2 = scipy.stats.t.pdf(highlight_x2, df.shape[0] - 1)

    #Critical alphas
    z_alphas = scipy.stats.t.ppf([alpha / 2, 1 - (alpha / 2)], df.shape[0] - 1)

    #Area traces
    area_trace1 = go.Line(x = highlight_x1, y = highlight_y1, fill = 'tozeroy', fillcolor = 'blue')
    area_trace2 = go.Line(x = highlight_x2, y = highlight_y2, fill = 'tozeroy', fillcolor = 'blue')

    #Create figure
    fig = go.Figure(data = [line_trace, area_trace1, area_trace2])

    #Add vertical lines for z-score and critical value
    fig.add_vline(-1 * t_score, line_dash = 'dash', line_color = 'blue')
    fig.add_vline(t_score, line_dash = 'dash', line_color = 'blue')
    fig.add_vline(z_alphas[0], line_dash = 'dash', line_color = 'red')
    fig.add_vline(z_alphas[1], line_dash = 'dash', line_color = 'red')

    #Hide legend and add tickmarks
    fig.update_layout(showlegend = False)
    fig.update_xaxes(tickvals = [i for i in range(-3, 4)] + [t_score] + list(z_alphas),
                    ticktext = [i for i in range(-3, 4)] + ['Test Statistic'] + ['Critical Value', 'Critical Value'])

else:

    if sided_test == 'Left-sided':

        #Values left of z-score
        highlight_x = np.arange(-3, t_score, (t_score + 3) / 1000)
        z_alpha = scipy.stats.t.ppf(alpha, df.shape[0] - 1)

    elif sided_test == 'Right-sided':
    
        #Values right of z-score
        highlight_x = np.arange(t_score, 3, (3 - t_score) / 1000)
        z_alpha = scipy.stats.t.ppf(1 - alpha, df.shape[0] - 1)

    #PDF of x-values
    highlight_y = scipy.stats.t.pdf(highlight_x, df.shape[0] - 1)

    #Area under z-score
    area_trace = go.Line(x = highlight_x, y = highlight_y, fill = 'tozeroy', fillcolor = 'blue')

    #Create figure
    fig = go.Figure(data = [line_trace, area_trace])

    #Vertical lines for z-score and critical value
    fig.add_vline(t_score, line_dash = 'dash', line_color = 'blue')
    fig.add_vline(z_alpha, line_dash = 'dash', line_color = 'red')

    #Hide legend and add tick marks
    fig.update_layout(showlegend = False)
    fig.update_xaxes(tickvals = [i for i in range(-3, 4)] + [t_score, z_alpha],
                    ticktext = [i for i in range(-3, 4)] + ['Test Statistic', 'Critical Value'])

#Show plot
col2.plotly_chart(fig, use_container_width = True)

###########
## Tab 2 ##
###########

#Expander of test parameters
with tab2.expander('Test parameters', expanded = True):

    sided_test = st.selectbox('Test Type', ['Left-sided', 'Right-sided', '2-sided'], key = 'paired_select')
    alpha = st.number_input('Significance level ($\\alpha$)', value = 0.05, key = 'paired_alpha')
    sample1 = st.selectbox('Sample 1', df.columns)
    sample2 = st.selectbox('Sample 2', df.columns, index = 1)

#Alternative hypotheses
h1s = {'Left-sided' : f'$H_1: \mu_1 < \mu_2$',
       'Right-sided' : f'$H_1: \mu_1 > \mu_2$',
       '2-sided' : f'$H_1: \mu_1 \\neq \mu_2$'}

tab2.divider()

col1, col2, col3 = tab2.columns([1, 2, 1])

#Write null and alternative hypothesis
col1.header('Hypotheses')
col1.write(f'$$H_0: \mu_1 = \mu_2$$')
col1.write(h1s[sided_test])

#Calculate mean
sample1_mean = df[sample1].mean()
sample1_std = df[sample1].std()
sample2_mean = df[sample2].mean()
sample2_std = df[sample2].std()

diff = df[sample1] - df[sample2]
sample_diff_mean = diff.mean()
sample_diff_std = diff.std()

#Write samples mean
col2.header('Sample Statistics')

subcol1, subcol2, subcol3 = col2.columns(3)
subcol1.write('$\overline{X}_1 = ' + str(round(sample1_mean, 4)) + '$')
subcol1.write('$S^2_1 = ' + str(round(sample1_std, 4)) + '$')
subcol2.write('$\overline{X}_2 = ' + str(round(sample2_mean, 4)) + '$')
subcol2.write('$S^2_2 = ' + str(round(sample2_std, 4)) + '$')
subcol3.write('$\overline{X}_D = ' + str(round(sample_diff_mean, 4)) + '$')
subcol3.write('$S^2_D = ' + str(round(sample_diff_std, 4)) + '$')

#Calculate z-score
t_score = (sample_diff_mean) / (sample_diff_std / np.sqrt(df.shape[0]))

col3.header('Test Statistic')
col3.write('$T = ' + str(round(t_score, 4)) + '$')

tab2.divider()

#Calculate p-value
if sided_test == 'Left-sided':

    p_value = scipy.stats.t.cdf(t_score, df.shape[0] - 1)

elif sided_test == 'Right-sided':

    p_value = 1 - scipy.stats.t.cdf(t_score, df.shape[0] - 1)

else:

    p_value = 2 * (1 - scipy.stats.t.cdf(abs(t_score), df.shape[0] - 1))

#Interpret results
if p_value < alpha:

    result = f'Since p-value $ = {round(float(p_value), 4)} < \\alpha = {alpha}$, we __reject the null hypothesis__ in favor of the alternative (${h1s[sided_test].split(":")[-1]})'

else:

    result = f'Since p-value $ = {round(float(p_value), 4)} \geq \\alpha = {alpha}$, we __fail to reject the null hypothesis__ that $\mu = {mu}$'

col1, col2 = tab2.columns([1, 2])

#Write conclusion
col1.header('Conclusion')
col1.write(result)

#Create x and y-axis for plot
x_axis = np.arange(-3, 3, step = 6 / 1000)
y_axis = scipy.stats.t.pdf(x_axis, df.shape[0] - 1)

#Plot normal PDF
line_trace = go.Line(x = x_axis, y = y_axis, fillcolor = 'blue')

if sided_test == '2-sided':

    #Area under z-score
    highlight_x1 = np.arange(np.abs(t_score), 3, (3 - np.abs(t_score)) / 1000)
    highlight_x2 = np.arange(-3, -1 * np.abs(t_score), (3 - np.abs(t_score)) / 1000)
    highlight_y1 = scipy.stats.t.pdf(highlight_x1, df.shape[0] - 1)
    highlight_y2 = scipy.stats.t.pdf(highlight_x2, df.shape[0] - 1)

    #Critical alphas
    z_alphas = scipy.stats.t.ppf([alpha / 2, 1 - (alpha / 2)], df.shape[0] - 1)

    #Area traces
    area_trace1 = go.Line(x = highlight_x1, y = highlight_y1, fill = 'tozeroy', fillcolor = 'blue')
    area_trace2 = go.Line(x = highlight_x2, y = highlight_y2, fill = 'tozeroy', fillcolor = 'blue')

    #Create figure
    fig = go.Figure(data = [line_trace, area_trace1, area_trace2])

    #Add vertical lines for z-score and critical value
    fig.add_vline(-1 * t_score, line_dash = 'dash', line_color = 'blue')
    fig.add_vline(t_score, line_dash = 'dash', line_color = 'blue')
    fig.add_vline(z_alphas[0], line_dash = 'dash', line_color = 'red')
    fig.add_vline(z_alphas[1], line_dash = 'dash', line_color = 'red')

    #Hide legend and add tickmarks
    fig.update_layout(showlegend = False)
    fig.update_xaxes(tickvals = [i for i in range(-3, 4)] + [t_score] + list(z_alphas),
                    ticktext = [i for i in range(-3, 4)] + ['Test Statistic'] + ['Critical Value', 'Critical Value'])

else:

    if sided_test == 'Left-sided':

        #Values left of z-score
        highlight_x = np.arange(-3, t_score, (t_score + 3) / 1000)
        z_alpha = scipy.stats.t.ppf(alpha, df.shape[0] - 1)

    elif sided_test == 'Right-sided':
    
        #Values right of z-score
        highlight_x = np.arange(t_score, 3, (3 - t_score) / 1000)
        z_alpha = scipy.stats.t.ppf(1 - alpha, df.shape[0] - 1)

    #PDF of x-values
    highlight_y = scipy.stats.t.pdf(highlight_x, df.shape[0] - 1)

    #Area under z-score
    area_trace = go.Line(x = highlight_x, y = highlight_y, fill = 'tozeroy', fillcolor = 'blue')

    #Create figure
    fig = go.Figure(data = [line_trace, area_trace])

    #Vertical lines for z-score and critical value
    fig.add_vline(t_score, line_dash = 'dash', line_color = 'blue')
    fig.add_vline(z_alpha, line_dash = 'dash', line_color = 'red')

    #Hide legend and add tick marks
    fig.update_layout(showlegend = False)
    fig.update_xaxes(tickvals = [i for i in range(-3, 4)] + [t_score, z_alpha],
                    ticktext = [i for i in range(-3, 4)] + ['Test Statistic', 'Critical Value'])

#Show plot
col2.plotly_chart(fig, use_container_width = True)

###########
## Tab 3 ##
###########

#Expander of test parameters
with tab3.expander('Test parameters', expanded = True):

    sided_test = st.selectbox('Test Type', ['Left-sided', 'Right-sided', '2-sided'], key = 'independent_select')
    alpha = st.number_input('Significance level ($\\alpha$)', value = 0.05, key = 'independent_alpha')
    equal_var = st.toggle('Assume Equal Variance')
    sample1 = st.selectbox('Sample 1', df.columns, key = 'independent_sample1')
    sample2 = st.selectbox('Sample 2', df.columns, index = 1, key = 'independent_sample2')

#Alternative hypotheses
h1s = {'Left-sided' : f'$H_1: \mu_1 < \mu_2$',
       'Right-sided' : f'$H_1: \mu_1 > \mu_2$',
       '2-sided' : f'$H_1: \mu_1 \\neq \mu_2$'}

tab3.divider()

col1, col2, col3 = tab3.columns(3)

#Write null and alternative hypothesis
col1.header('Hypotheses')
col1.write(f'$$H_0: \mu_1 = \mu_2$$')
col1.write(h1s[sided_test])

#Calculate mean
sample1_mean = df[sample1].mean()
sample1_std = df[sample1].std()
sample2_mean = df[sample2].mean()
sample2_std = df[sample2].std()
sample1_n = len(df[sample1].dropna())
sample2_n = len(df[sample2].dropna())

#Write samples mean
col2.header('Sample Statistics')

subcol1, subcol2 = col2.columns(2)
subcol1.write('$\overline{X}_1 = ' + str(round(sample1_mean, 4)) + '$')
subcol1.write('$S^2_1 = ' + str(round(sample1_std, 4)) + '$')
subcol1.write('$n_1 = ' + str(sample1_n) + '$')
subcol2.write('$\overline{X}_2 = ' + str(round(sample2_mean, 4)) + '$')
subcol2.write('$S^2_2 = ' + str(round(sample2_std, 4)) + '$')
subcol2.write('$n_2 = ' + str(sample2_n) + '$')

if equal_var:

    #Calculate t-score
    pooled_variance = np.sqrt(((sample1_n - 1) * (sample1_std ** 2) + (sample2_n - 1) * (sample2_std ** 2)) / sample1_n + sample2_n - 2)
    t_score = (sample1_mean - sample2_mean) / (pooled_variance * np.sqrt((1 / sample1_n) + (1 / sample2_n)))

    dof = sample1_n + sample2_n - 2

else:

    #Calculate z-score
    s_delta = np.sqrt(((sample1_std ** 2) / sample1_n) + ((sample2_std ** 2) / sample2_n))
    t_score = (sample1_mean - sample2_mean) / s_delta

    numerator = (s_delta ** 2) ** 2
    denominator = (((sample1_std ** 2) / sample1_n) ** 2 / (sample1_n - 1) + ((sample2_std ** 2) / sample2_n) ** 2 / (sample2_n - 1))

    dof = numerator / denominator

col3.header('Test Statistic')
col3.write('$T = ' + str(round(t_score, 4)) + '$')
col3.write('$d.o.f = ' + str(round(dof, 4)) + '$')

tab3.divider()

#Calculate p-value
if sided_test == 'Left-sided':

    p_value = scipy.stats.t.cdf(t_score, dof)

elif sided_test == 'Right-sided':

    p_value = 1 - scipy.stats.t.cdf(t_score, dof)

else:

    p_value = 2 * (1 - scipy.stats.t.cdf(abs(t_score), dof))

#Interpret results
if p_value < alpha:

    result = f'Since p-value $ = {round(float(p_value), 4)} < \\alpha = {alpha}$, we __reject the null hypothesis__ in favor of the alternative (${h1s[sided_test].split(":")[-1]})'

else:

    result = f'Since p-value $ = {round(float(p_value), 4)} \geq \\alpha = {alpha}$, we __fail to reject the null hypothesis__ that $\mu = {mu}$'

col1, col2 = tab3.columns([1, 2])

#Write conclusion
col1.header('Conclusion')
col1.write(result)

#Create x and y-axis for plot
x_axis = np.arange(-3, 3, step = 6 / 1000)
y_axis = scipy.stats.t.pdf(x_axis, dof)

#Plot normal PDF
line_trace = go.Line(x = x_axis, y = y_axis, fillcolor = 'blue')

if sided_test == '2-sided':

    #Area under z-score
    highlight_x1 = np.arange(np.abs(t_score), 3, (3 - np.abs(t_score)) / 1000)
    highlight_x2 = np.arange(-3, -1 * np.abs(t_score), (3 - np.abs(t_score)) / 1000)
    highlight_y1 = scipy.stats.t.pdf(highlight_x1, dof)
    highlight_y2 = scipy.stats.t.pdf(highlight_x2, dof)

    #Critical alphas
    z_alphas = scipy.stats.t.ppf([alpha / 2, 1 - (alpha / 2)], dof)

    #Area traces
    area_trace1 = go.Line(x = highlight_x1, y = highlight_y1, fill = 'tozeroy', fillcolor = 'blue')
    area_trace2 = go.Line(x = highlight_x2, y = highlight_y2, fill = 'tozeroy', fillcolor = 'blue')

    #Create figure
    fig = go.Figure(data = [line_trace, area_trace1, area_trace2])

    #Add vertical lines for z-score and critical value
    fig.add_vline(-1 * t_score, line_dash = 'dash', line_color = 'blue')
    fig.add_vline(t_score, line_dash = 'dash', line_color = 'blue')
    fig.add_vline(z_alphas[0], line_dash = 'dash', line_color = 'red')
    fig.add_vline(z_alphas[1], line_dash = 'dash', line_color = 'red')

    #Hide legend and add tickmarks
    fig.update_layout(showlegend = False)
    fig.update_xaxes(tickvals = [i for i in range(-3, 4)] + [t_score] + list(z_alphas),
                    ticktext = [i for i in range(-3, 4)] + ['Test Statistic'] + ['Critical Value', 'Critical Value'])

else:

    if sided_test == 'Left-sided':

        #Values left of z-score
        highlight_x = np.arange(-3, t_score, (t_score + 3) / 1000)
        z_alpha = scipy.stats.t.ppf(alpha, dof)

    elif sided_test == 'Right-sided':
    
        #Values right of z-score
        highlight_x = np.arange(t_score, 3, (3 - t_score) / 1000)
        z_alpha = scipy.stats.t.ppf(1 - alpha, dof)

    #PDF of x-values
    highlight_y = scipy.stats.t.pdf(highlight_x, dof)

    #Area under z-score
    area_trace = go.Line(x = highlight_x, y = highlight_y, fill = 'tozeroy', fillcolor = 'blue')

    #Create figure
    fig = go.Figure(data = [line_trace, area_trace])

    #Vertical lines for z-score and critical value
    fig.add_vline(t_score, line_dash = 'dash', line_color = 'blue')
    fig.add_vline(z_alpha, line_dash = 'dash', line_color = 'red')

    #Hide legend and add tick marks
    fig.update_layout(showlegend = False)
    fig.update_xaxes(tickvals = [i for i in range(-3, 4)] + [t_score, z_alpha],
                    ticktext = [i for i in range(-3, 4)] + ['Test Statistic', 'Critical Value'])

#Show plot
col2.plotly_chart(fig, use_container_width = True)