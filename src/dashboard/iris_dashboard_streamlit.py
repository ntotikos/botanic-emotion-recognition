"""Testing streamlit here. """

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import time
#plotly.express as px

# Streamlit page setup
st.set_page_config(
    page_title="Real-Time Data Science Dashboard",
    page_icon="✅",
    layout="wide",
)

# Get data.
dataset_url = "https://raw.githubusercontent.com/Lexie88rus/bank-marketing-analysis/master/bank.csv"

# read csv from a URL
@st.cache_data
def get_data() -> pd.DataFrame:
    return pd.read_csv(dataset_url)


df = get_data()

# Load the dataset
iris = load_iris()
df = pd.DataFrame(data= np.c_[iris['data'], iris['target']], columns= iris['feature_names'] + ['target'])
print(df.columns.tolist())

# Title
st.title("Iris Dataset Explorer")

# Introduction
st.write("""
This is a simple Streamlit dashboard that allows you to explore the Iris dataset.
Select the visualization types from the sidebar.
""")

sepal_length_filter = st.selectbox("Select the Job", pd.unique(df['sepal length (cm)']))
df = df[df['sepal length (cm)'] == sepal_length_filter]

# create three columns
kpi1, kpi2, kpi3 = st.columns(3)
avg_age = 33
count_married = 12
balance = 2
# fill in those three columns with respective metrics or KPIs
kpi1.metric(
    label="Age ⏳",
    value=round(avg_age),
    delta=round(avg_age) - 10,
)

kpi2.metric(
    label="Married Count 💍",
    value=int(count_married),
    delta=-10 + count_married,
)

kpi3.metric(
    label="A/C Balance ＄",
    value=f"$ {round(balance,2)} ",
    delta=-round(balance / count_married) * 100,
)

x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.figure(figsize=(10, 5))
plt.plot(x, y)
plt.title('Sin Wave using Matplotlib')
plt.xlabel('X Axis')
plt.ylabel('Y Axis')

st.pyplot(plt)

# create two columns for charts
fig_col1, fig_col2 = st.columns(2)

with fig_col1:
    st.markdown("### First Chart")
    st.pyplot(plt)

with fig_col2:
    st.markdown("### Second Chart")
    st.pyplot(plt)

st.markdown("### Detailed Data View")
st.dataframe(df)


#for seconds in range(200):

#    df['new sepal length (cm)'] = df['sepal length (cm)'] * np.random.choice(range(1, 5))
#    df['new petal width (cm)'] = df['petal width (cm)'] * np.random.choice(range(1, 5))
#    time.sleep(1)



