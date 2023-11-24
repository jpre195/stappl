import streamlit as st
import pandas as pd
import scipy
import numpy as np

st.title('Statistical Testing APPLication')

st.write('This is the homepage')

#Upload a file
file = st.file_uploader('Choose a file', type = ['csv'])

if file is not None:

    df = pd.read_csv(file)

    st.session_state['df'] = df

    st.dataframe(df.head(10), use_container_width = True, hide_index = True)