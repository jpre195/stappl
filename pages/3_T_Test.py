import streamlit as st
import pandas as pd
import scipy
import numpy as np
import plotly.graph_objects as go

#Wide mode
st.set_page_config(layout = 'wide')

#Title of page
st.title('T-Test')

#Upload a file
file = st.file_uploader('Choose a file', type = ['csv'])

#Expander of test parameters
with st.expander('Test parameters', expanded = True):

    sided_test = st.selectbox('Test Type', ['Left-sided', 'Right-sided', '2-sided'])
    mu = st.number_input('Population mean ($\mu_0$)')
    alpha = st.number_input('Significance level ($\\alpha$)', value = 0.05)

#Alternative hypotheses
h1s = {'Left-sided' : f'$H_1: \mu < {mu}$',
       'Right-sided' : f'$H_1: \mu > {mu}$',
       '2-sided' : f'$H_1: \mu \\neq {mu}$'}

#If file has been uploaded
if file is not None:

    #Read CSV file
    df = pd.DataFrame() if file is None else pd.read_csv(file)

    st.divider()

    col1, col2 = st.columns(2)

    #Write null and alternative hypothesis
    col1.header('Hypotheses')
    col1.write(f'$$H_0: \mu = {mu}$$')
    col1.write(h1s[sided_test])

    #Calculate mean
    sample_mean = df.mean()
    # sample_std = np.sqrt(df.var())
    sample_std = df.std()

    #Write samples mean
    col2.header('Sample Statistics')
    col2.write('$\overline{X} = ' + str(round(sample_mean.values[0], 4)) + '$')

    st.divider()

    #Calculate z-score
    t_score = (sample_mean - mu) / (sample_std / np.sqrt(df.shape[0]))
    t_score = list(t_score)[0]

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

    col1, col2 = st.columns([1, 2])

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