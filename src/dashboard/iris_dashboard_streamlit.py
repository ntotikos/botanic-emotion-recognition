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

# Title
st.title("Iris Dataset Explorer")

# Introduction
st.write("""
This is a simple Streamlit dashboard that allows you to explore the Iris dataset.
Select the visualization types from the sidebar.
""")

sepal_length = st.selectbox("Select the Job", pd.unique(df['sepal length (cm)']))

x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.figure(figsize=(10, 5))
plt.plot(x, y)
plt.title('Sin Wave using Matplotlib')
plt.xlabel('X Axis')
plt.ylabel('Y Axis')

st.pyplot(plt)
