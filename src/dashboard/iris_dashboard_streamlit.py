"""Testing streamlit here. """

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

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

x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.figure(figsize=(10, 5))
plt.plot(x, y)
plt.title('Sin Wave using Matplotlib')
plt.xlabel('X Axis')
plt.ylabel('Y Axis')

st.pyplot(plt)
