import streamlit as st
from annoy import AnnoyIndex
from sentence_transformers import SentenceTransformer
import pandas as pd
# from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode
import glob
import spacy
from spacy.cli import download
from collections import Counter
import seaborn as sns

import plotly.express as px

st.set_page_config(layout="wide")
@st.cache_resource()
def load_annoy():
    t = AnnoyIndex(384, "angular")
    t.load("annoy_index.ann")
    return t

@st.cache_resource()
def load_model():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model

@st.cache_data()
def load_data():
    df = pd.read_csv("database.csv")
    return df

@st.cache_resource()
def load_nlp():
    model_name = "en_core_web_sm"
    try:
        nlp = spacy.load(model_name, disable=["ner"])
    except:
        OSError
        print(f"Downloading {model_name}...")
        try:
            download(model_name)
            nlp = spacy.load(model_name, disable=["ner"])
        except:
            OSError
            raise Exception(f"{model_name} is not a recognized spaCy model.")
    return nlp

def create_plot(df, word_cat, frequency_cat, pal_color, n=10):
    pal = list(sns.color_palette(palette=pal_color, n_colors=n).as_hex())
    print(pal)
    fig = px.pie(df[0:n], values=frequency_cat, names=word_cat,
             color_discrete_sequence=pal)
    print(fig)
    fig.update_traces(textposition='outside', textinfo='percent+label', 
                    hole=.6, hoverinfo="label+percent+name")

    fig.update_layout(width = 500, height = 600,
                    margin = dict(t=0, l=0, r=0, b=0))
    return fig

def token_counter(tokens):
    token_count = Counter(tokens)
    token_df = pd.DataFrame(token_count.most_common(), columns=["word", "count"])
    return token_df



files = glob.glob("./data/*.json")
files.sort()

t = load_annoy()
model = load_model()
df = load_data()

nlp = load_nlp()

st.title("USHMM Testimony Semantic Search")

query = st.text_input("Input Query Text")

sel_col1, sel_col2 = st.columns(2)
n = sel_col1.slider("Number of Words to Analyze", 10, 100)
num_results = sel_col2.slider("Number of Results", 10, 100)
if query:
    emb = model.encode(query)
    res = t.get_nns_by_vector(emb, num_results)
    res = df.iloc[res]
    texts = res.text.tolist()
    adjs = []
    verbs = []
    nouns = []
    verb_stops = ["have", "be", "go", "get", "take",
                "say", "know", "do", "call", "ask",
                "come", "think", "make", "happen"]
    for text in texts:
        doc = nlp(text)
        for token in doc:
            if token.pos_ =="ADJ":
                adjs.append(token.lemma_)
            elif token.pos_ =="VERB":
                if token.lemma_ not in verb_stops:
                    verbs.append(token.lemma_)
            elif token.pos_ == "NOUN":
                if len(token.lemma_) > 2:
                    nouns.append(token.lemma_)
    
    adj_df = token_counter(adjs)
    verb_df = token_counter(verbs)
    noun_df = token_counter(nouns)

    adj_plot = create_plot(adj_df, "word", "count", "YlGn_r", n=n)
    verb_plot = create_plot(verb_df, "word", "count", "Reds_r", n=n)
    noun_plot = create_plot(noun_df, "word", "count", "Purples_r", n=n)

    data_expander = st.expander("Expand for Data Visualizations")
    col1, col2 = data_expander.columns(2)

    col1.markdown("<center><h2>Verbs</h2></center>", unsafe_allow_html=True)
    col1.plotly_chart(verb_plot)

    col2.markdown("<center><h2>Adjectives</h2></center>", unsafe_allow_html=True)
    col2.plotly_chart(adj_plot)

    col1.markdown("<center><h2>Nouns</h2></center>", unsafe_allow_html=True)
    col1.plotly_chart(noun_plot)
    # st.write(verb_df)
    st.markdown(res.to_markdown())
    # AgGrid(df.iloc[res], fit_columns_on_grid_load=True)