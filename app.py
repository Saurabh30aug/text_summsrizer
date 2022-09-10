#!/usr/bin/env python
# coding: utf-8

# In[1]:
import validators, re
import torch
from fake_useragent import UserAgent
from bs4 import BeautifulSoup   
import streamlit as st
from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer
from sentence_transformers import SentenceTransformer
import en_core_web_lg
import time
import base64 
import requests
import docx2txt
from io import StringIO
from PyPDF2 import PdfFileReader
import warnings
import nltk
import itertools
import numpy as np

nltk.download('punkt')

from nltk import sent_tokenize

warnings.filterwarnings("ignore")


# In[2]:

time_str = time.strftime("%d%m%Y-%H%M%S")
HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem; 
margin-bottom: 2.5rem">{}</div> """
#Functions

def article_text_extractor(url: str):
    
    '''Extract text from url and divide text into chunks if length of text is more than 500 words'''
    
    ua = UserAgent()

    headers = {'User-Agent':str(ua.chrome)}

    r = requests.get(url,headers=headers)
    
    soup = BeautifulSoup(r.text, "html.parser")
    title_text = soup.find_all(["h1"])
    para_text = soup.find_all(["p"])
    article_text = [result.text for result in para_text]
    
    try:
    
        article_header = [result.text for result in title_text][0]
        
    except:
    
        article_header = ''
        
    article = nlp(" ".join(article_text))
    sentences = [i.text for i in list(article.sents)]
    
    current_chunk = 0
    chunks = []
    
    for sentence in sentences:
        if len(chunks) == current_chunk + 1:
            if len(chunks[current_chunk]) + len(sentence.split(" ")) <= 500:
                chunks[current_chunk].extend(sentence.split(" "))
            else:
                current_chunk += 1
                chunks.append(sentence.split(" "))
        else:
            chunks.append(sentence.split(" "))

    for chunk_id in range(len(chunks)):
        chunks[chunk_id] = " ".join(chunks[chunk_id])

    return article_header, chunks
 
def chunk_clean_text(text):
    
    """Chunk text longer than 500 tokens"""
    
    article = nlp(text)
    sentences = [i.text for i in list(article.sents)]
    
    current_chunk = 0
    chunks = []
    
    for sentence in sentences:
        if len(chunks) == current_chunk + 1:
            if len(chunks[current_chunk]) + len(sentence.split(" ")) <= 500:
                chunks[current_chunk].extend(sentence.split(" "))
            else:
                current_chunk += 1
                chunks.append(sentence.split(" "))
        else:
            chunks.append(sentence.split(" "))

    for chunk_id in range(len(chunks)):
        chunks[chunk_id] = " ".join(chunks[chunk_id])
    
    return chunks
    
def preprocess_plain_text(x):

    x = x.encode("ascii", "ignore").decode()  # unicode
    x = re.sub(r"https*\S+", " ", x)  # url
    x = re.sub(r"@\S+", " ", x)  # mentions
    x = re.sub(r"#\S+", " ", x)  # hastags
    x = re.sub(r"\s{2,}", " ", x)  # over spaces
    x = re.sub("[^.,!?A-Za-z0-9]+", " ", x)  # special charachters except .,!?

    return x

def extract_pdf(file):
    
    '''Extract text from PDF file'''
    
    pdfReader = PdfFileReader(file)
    count = pdfReader.numPages
    all_text = ""
    for i in range(count):
        page = pdfReader.getPage(i)
        all_text += page.extractText()
    

    return all_text


def extract_text_from_file(file):
    
    '''Extract text from uploaded file'''

    # read text file
    if file.type == "text/plain":
        # To convert to a string based IO:
        stringio = StringIO(file.getvalue().decode("utf-8"))

        # To read file as string:
        file_text = stringio.read()

    # read pdf file
    elif file.type == "application/pdf":
        file_text = extract_pdf(file)

    # read docx file
    elif (
        file.type
        == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ):
        file_text = docx2txt.process(file)

    return file_text

def summary_downloader(raw_text):
    
	b64 = base64.b64encode(raw_text.encode()).decode()
	new_filename = "new_text_file_{}_.txt".format(time_str)
	st.markdown("#### Download Summary as a File ###")
	href = f'<a href="data:file/txt;base64,{b64}" download="{new_filename}">Click to Download!!</a>'
	st.markdown(href,unsafe_allow_html=True)
	
def get_all_entities_per_sentence(text):
    doc = nlp(''.join(text))

    sentences = list(doc.sents)

    entities_all_sentences = []
    for sentence in sentences:
        entities_this_sentence = []

        # SPACY ENTITIES
        for entity in sentence.ents:
            entities_this_sentence.append(str(entity))

        # FLAIR ENTITIES (CURRENTLY NOT USED)
        # sentence_entities = Sentence(str(sentence))
        # tagger.predict(sentence_entities)
        # for entity in sentence_entities.get_spans('ner'):
        #     entities_this_sentence.append(entity.text)

        # XLM ENTITIES
        entities_xlm = [entity["word"] for entity in ner_model(str(sentence))]
        for entity in entities_xlm:
            entities_this_sentence.append(str(entity))

        entities_all_sentences.append(entities_this_sentence)

    return entities_all_sentences

def get_all_entities(text):
    all_entities_per_sentence = get_all_entities_per_sentence(text)
    return list(itertools.chain.from_iterable(all_entities_per_sentence))
    
def get_and_compare_entities(article_content,summary_output):
    
    all_entities_per_sentence = get_all_entities_per_sentence(article_content)
    entities_article = list(itertools.chain.from_iterable(all_entities_per_sentence))
   
    all_entities_per_sentence = get_all_entities_per_sentence(summary_output)
    entities_summary = list(itertools.chain.from_iterable(all_entities_per_sentence))
   
    matched_entities = []
    unmatched_entities = []
    for entity in entities_summary:
        if any(entity.lower() in substring_entity.lower() for substring_entity in entities_article):
            matched_entities.append(entity)
        elif any(
                np.inner(sentence_embedding_model.encode(entity, show_progress_bar=False),
                         sentence_embedding_model.encode(art_entity, show_progress_bar=False)) > 0.9 for
                art_entity in entities_article):
            matched_entities.append(entity)
        else:
            unmatched_entities.append(entity)

    matched_entities = list(dict.fromkeys(matched_entities))
    unmatched_entities = list(dict.fromkeys(unmatched_entities))

    matched_entities_to_remove = []
    unmatched_entities_to_remove = []

    for entity in matched_entities:
        for substring_entity in matched_entities:
            if entity != substring_entity and entity.lower() in substring_entity.lower():
                matched_entities_to_remove.append(entity)

    for entity in unmatched_entities:
        for substring_entity in unmatched_entities:
            if entity != substring_entity and entity.lower() in substring_entity.lower():
                unmatched_entities_to_remove.append(entity)

    matched_entities_to_remove = list(dict.fromkeys(matched_entities_to_remove))
    unmatched_entities_to_remove = list(dict.fromkeys(unmatched_entities_to_remove))

    for entity in matched_entities_to_remove:
        matched_entities.remove(entity)
    for entity in unmatched_entities_to_remove:
        unmatched_entities.remove(entity)

    return matched_entities, unmatched_entities

def highlight_entities(article_content,summary_output):
   
    markdown_start_red = "<mark class=\"entity\" style=\"background: rgb(238, 135, 135);\">"
    markdown_start_green = "<mark class=\"entity\" style=\"background: rgb(121, 236, 121);\">"
    markdown_end = "</mark>"

    matched_entities, unmatched_entities = get_and_compare_entities(article_content,summary_output)
    
    print(summary_output)

    for entity in matched_entities:
        summary_output = re.sub(f'({entity})(?![^rgb\(]*\))',markdown_start_green + entity + markdown_end,summary_output)

    for entity in unmatched_entities:
        summary_output = re.sub(f'({entity})(?![^rgb\(]*\))',markdown_start_red + entity + markdown_end,summary_output)
    
    print("")
    print(summary_output)
    
    print("")
    print(summary_output)
    
    soup = BeautifulSoup(summary_output, features="html.parser")

    return HTML_WRAPPER.format(soup)


def clean_text(text,doc=False,plain_text=False,url=False):
    """Return clean text from the various input sources"""

    if url:
        is_url = validators.url(text)
        
        if is_url:
            # complete text, chunks to summarize (list of sentences for long docs)
            article_title,chunks = article_text_extractor(url=url_text)
        
            return article_title, chunks
        
    elif doc:
        
       clean_text = chunk_clean_text(preprocess_plain_text(extract_text_from_file(text)))
       
       return None, clean_text
    
    elif plain_text:
        
        clean_text = chunk_clean_text(preprocess_plain_text(text))
        
        return None, clean_text
        

@st.experimental_singleton(suppress_st_warning=True)
def get_spacy():
    nlp = en_core_web_lg.load()
    return nlp
    
@st.experimental_singleton(suppress_st_warning=True)
def facebook_model():
    model_name = 'facebook/bart-large-cnn'
    summarizer = pipeline('summarization',model=model_name,tokenizer=model_name,
    device=0 if torch.cuda.is_available() else -1)
    return summarizer
    
@st.experimental_singleton(suppress_st_warning=True)
def schleifer_model():
    model_name = 'sshleifer/distilbart-cnn-12-6'
    summarizer = pipeline('summarization',model=model_name, tokenizer=model_name,
    device=0 if torch.cuda.is_available() else -1)
    return summarizer

@st.experimental_singleton(suppress_st_warning=True)    
def google_model():
    model_name = 'google/pegasus-large'
    summarizer = pipeline('summarization',model=model_name, tokenizer=model_name,
    device=0 if torch.cuda.is_available() else -1)
    return summarizer
    
@st.experimental_singleton(suppress_st_warning=True)
def get_sentence_embedding_model():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
@st.experimental_singleton(suppress_st_warning=True)
def get_ner_pipeline():
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large-finetuned-conll03-english")
    model = AutoModelForTokenClassification.from_pretrained("xlm-roberta-large-finetuned-conll03-english")
    return pipeline("ner", model=model, tokenizer=tokenizer, grouped_entities=True)

# Load all different models (cached) at start time of the hugginface space
sentence_embedding_model = get_sentence_embedding_model()
ner_model = get_ner_pipeline()
nlp = get_spacy()
    
#Streamlit App
    
st.title("Article Text and Link Extractive Summarizer with Entity Matching 📝")

model_type = st.sidebar.selectbox(
    "Model type", options=["Facebook-Bart", "Sshleifer-DistilBart","Google-Pegasus"]
)

max_len= st.sidebar.slider("Maximum length of the summarized text",min_value=100,max_value=500,step=10)
min_len= st.sidebar.slider("Minimum length of the summarized text",min_value=50,max_value=200,step=10)

st.markdown(
    "Model Source: [Facebook-Bart-large-CNN](https://huggingface.co/facebook/bart-large-cnn), [Sshleifer-distilbart-cnn-12-6](https://huggingface.co/sshleifer/distilbart-cnn-12-6) and [Google-Pegasus-large](https://huggingface.co/google/pegasus-large)"
)

st.markdown(
    """The app supports extractive summarization which aims to identify the salient information that is then extracted and grouped together to form a concise summary. 
    For documents or text that is more than 500 words long, the app will divide the text into chunks and summarize each chunk. Please note when using the sidebar slider, those values represent the min/max text length per chunk of text to be summarized. If your article to be summarized is 1000 words, it will be divided into two chunks of 500 words first then the default max length of 100 words is applied per chunk, resulting in a summarized text with 200 words maximum. 
    There are two models available to choose from:""")

st.markdown("""   
    - Facebook-Bart, trained on large [CNN and Daily Mail](https://huggingface.co/datasets/cnn_dailymail) news articles.
    - Sshleifer-Distilbart, which is a distilled (smaller) version of the large Bart model.
    - Google Pegasus, trained on large C4 and HugeNews articles"""
)

st.markdown("""Please do note that the model will take longer to generate summaries for documents that are too long.""")

st.markdown(
    "The app only ingests the below formats for summarization task:"
)
st.markdown(
    """- Raw text entered in text box. 
- URL of an article to be summarized. 
- Documents with .txt, .pdf or .docx file formats."""
)

st.markdown("---")

if "text_area" not in st.session_state:
    st.session_state.text_area = ''
    
if "summ_area" not in st.session_state:
    st.session_state.summ_area = ''
        
url_text = st.text_input("Please Enter a url here")

st.markdown(
    "<h3 style='text-align: center; color: red;'>OR</h3>",
    unsafe_allow_html=True,
)

plain_text = st.text_area("Please Paste/Enter plain text here",)

st.markdown(
    "<h3 style='text-align: center; color: red;'>OR</h3>",
    unsafe_allow_html=True,
)

upload_doc = st.file_uploader(
    "Upload a .txt, .pdf, .docx file for summarization"
)

if url_text:
    article_title, cleaned_text = clean_text(url_text, url=True)
    st.session_state.text_area = cleaned_text[0]
    
elif plain_text:
    article_title, cleaned_text = clean_text(plain_text,plain_text=True)
    st.session_state.text_area = ''.join(cleaned_text)

elif upload_doc:
   article_title, cleaned_text = clean_text(upload_doc,doc=True)
   st.session_state.text_area = ''.join(cleaned_text)
   
article_text = st.text_area(
    label='Full Article Text',
    placeholder="Full article text will be displayed here..",
    height=250,
    key='text_area'
)
    
summarize = st.button("Summarize")

# called on toggle button [summarize]
if summarize:
    if model_type == "Facebook-Bart":
        if url_text:
            text_to_summarize =cleaned_text[0]
        else:
            text_to_summarize = cleaned_text

        with st.spinner(
            text="Loading Facebook-Bart Model and Extracting summary. This might take a few seconds depending on the length of your text..."
        ):
            summarizer_model = facebook_model()
            summarized_text = summarizer_model(text_to_summarize, max_length=max_len, min_length=min_len,clean_up_tokenization_spaces=True,no_repeat_ngram_size=4)
            summarized_text = ' '.join([summ['summary_text'] for summ in summarized_text])
           
    
    elif model_type == "Sshleifer-DistilBart":
        if url_text:
            text_to_summarize = cleaned_text[0]
        else:
            text_to_summarize = cleaned_text

        with st.spinner(
            text="Loading Sshleifer-DistilBart Model and Extracting summary. This might take a few seconds depending on the length of your text..."
        ):
            summarizer_model = schleifer_model()
            summarized_text = summarizer_model(text_to_summarize, max_length=max_len, min_length=min_len,clean_up_tokenization_spaces=True,no_repeat_ngram_size=4)
            summarized_text = ' '.join([summ['summary_text'] for summ in summarized_text])
            
    elif model_type == "Google-Pegasus":
        if url_text:
            text_to_summarize = cleaned_text[0]
            
        else:
            text_to_summarize = cleaned_text

        with st.spinner(
            text="Loading Google-Pegasus Model and Extracting summary. This might take a few seconds depending on the length of your text..."
        ):
            summarizer_model = google_model()
            summarized_text = summarizer_model(text_to_summarize, max_length=max_len, min_length=min_len,clean_up_tokenization_spaces=True,no_repeat_ngram_size=4)
            summarized_text = ' '.join([summ['summary_text'] for summ in summarized_text])
    
    with st.spinner("Calculating and matching entities, this takes a few seconds..."):
        entity_match_html = highlight_entities(text_to_summarize,summarized_text)
        st.markdown("####")
        print(entity_match_html)
        
        if article_title:

            # view summarized text (expander)
            st.markdown(f"Article title: {article_title}")
        
        st.session_state.summ_area = summarized_text
        
        st.subheader('Summarized Text with no Entity Matching')
        
        summarized_text = st.text_area(
        label = '',
        placeholder="Full summarized text will be displayed here..",
        height=250,
        key='summ_area'
        )
        
        st.markdown("####")     
        
        st.subheader("Summarized text with matched entities in Green and mismatched entities in Red relative to the Original Text")
   
        st.write(entity_match_html, unsafe_allow_html=True)
        
        st.markdown("####")     
    
        summary_downloader(summarized_text)


st.markdown("""
            """)
                        
#st.markdown("![visitor badge](https://visitor-badge.glitch.me/badge?page_id=nickmuchi-article-text-summarizer)")
# In[ ]:



