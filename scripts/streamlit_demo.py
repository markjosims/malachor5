import streamlit as st
import pandas as pd
import altair as alt
from streamlit_vega_lite import altair_component
from playsound import playsound
import os

st.title("Penguin Data Explorer 🐧")

st.write("Hover over the scatterplot to reveal details about a penguin. The code for this demo is at https://github.com/domoritz/streamlit-vega-lite-demo.")

@st.cache
def load(url):
    return  pd.read_csv(url)

df = load("/Users/markjos/projects/malachor5/data/embeddings.csv")

if st.checkbox("Show Raw Data"):
    st.write(df)

@st.cache
def make_altair_scatterplot():
    selected = alt.selection_single(on="click", empty="none")

    return alt.Chart(df).mark_circle(size=150).encode(
        alt.X("ecapa_voxlingua_x1", scale=alt.Scale(zero=False)),
        alt.Y("ecapa_voxlingua_x2", scale=alt.Scale(zero=False)),
        # color=alt.condition(selected, alt.value("red"), alt.value("steelblue"))
        color=alt.Color("lang")
    ).add_selection(selected)


selection = altair_component(make_altair_scatterplot())

if "_vgsid_" in selection:
    # the ids start at 1
    id=selection["_vgsid_"][0] - 1
    row = df.iloc[[id]]
    st.write(row)
    audiopath = row['audio'].item()
    audiopath = os.path.abspath(audiopath)
    playsound(audiopath)

else:
    st.info("Hover over the chart above to see details about the Penguin here.")