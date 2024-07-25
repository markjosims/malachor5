import streamlit as st
import pandas as pd
import altair as alt
from streamlit_vega_lite import altair_component
from playsound import playsound
import os

@st.cache
def load(url):
    return  pd.read_csv(url)

df = load("/Users/markjos/projects/malachor5/data/embeddings.csv")

if st.checkbox("Show Raw Data"):
    st.write(df)

@st.cache
def make_altair_histo():
    #selected = alt.selection_interval(encodings=['x'])
    
    histo=alt.Chart(df).mark_bar().encode(
        x=alt.X("sec:Q", bin=True),
        y="count()",
        color=alt.Color("lang")
    )#.add_selection(selected)

    return histo#, selected

@st.cache
def make_altair_scatterplot():
    clicked = alt.selection_single(on="click", empty="none")
    scatter=alt.Chart(df).mark_circle(size=150).encode(
        alt.X("ecapa_voxlingua_x1", scale=alt.Scale(zero=False)),
        alt.Y("ecapa_voxlingua_x2", scale=alt.Scale(zero=False)),
        size="sec",
        color=alt.Color("lang")
    ).add_selection(clicked)

    return scatter

histo = make_altair_histo()
scatter = make_altair_scatterplot()
altair_component(histo)
selection = altair_component(scatter)

st.info(selection)
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