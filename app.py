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
import json
import plotly.express as px
import gdown
import os

def download_file():
    # url = "https://drive.google.com/drive/folders/1P7LKV8ooTZn0ltLpHZifieAyGF5Q6oIL"
    url = "https://drive.google.com/drive/folders/158bJisQ3qOhTl3qn2poXwgFtXHzc44yJ"
    gdown.download_folder(url, quiet=True, use_cookies=False)
    print('index downloaded')

st.set_page_config(layout="wide")
@st.cache_resource()
def load_annoy():
    if os.path.isfile("ushmm/official_index.ann"):
        pass
    else:
        download_file()
    t = AnnoyIndex(384, "angular")
    t.load("ushmm/official_index.ann")
    return t

@st.cache_resource()
def load_model():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model

@st.cache_data()
def load_data():
    df = pd.read_csv("database.csv")
    rgs = df.rg.unique()
    rgs.sort()

    return df, rgs

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
    # print(pal)
    fig = px.pie(df[0:n], values=frequency_cat, names=word_cat,
             color_discrete_sequence=pal)
    # print(fig)
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
df, rgs = load_data()

nlp = load_nlp()

st.title("USHMM Testimony Semantic Search")

mode = st.selectbox("Select Mode", ["Semantic Search Engine", "Read a Testimony"])

if mode == "Semantic Search Engine":

    query = st.text_input("Input Query Text")

    sel_col1, sel_col2 = st.columns(2)
    word_analysis = sel_col1.checkbox("Analyze Words?")
    if word_analysis:
        n = sel_col1.slider("Number of Words to Analyze", 10, 100)
    num_results = sel_col1.slider("Number of Results", 10, 100)
    if query:
        emb = model.encode(query)
        res = t.get_nns_by_vector(emb, num_results)
        res = df.iloc[res]
        if word_analysis:
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
        st.table(res)
        # AgGrid(df.iloc[res], fit_columns_on_grid_load=True)
else:
    
    testimony = st.selectbox("Select Testimony", rgs)
    with open(f"data/{testimony}_trs_en.json" , "r") as f:
        data = json.load(f)
    res = pd.DataFrame(data["sequence"], columns=["Dialogue"])

    # res = df.loc[df["rg"] == testimony]
    st.table(res)
    # st.write(res)
