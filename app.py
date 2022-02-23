import streamlit as st
from gensim.summarization import summarize
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import textstat
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from pysummarization.nlpbase.auto_abstractor import AutoAbstractor
from pysummarization.tokenizabledoc.simple_tokenizer import SimpleTokenizer
from pysummarization.abstractabledoc.top_n_rank_abstractor import TopNRankAbstractor

@st.cache (persist = True)
def download ():
    nltk.download('punkt')
    nltk.download('stopwords')

download()

st.title('Text Summariser')

st.sidebar.title('How It Works')
st.sidebar.markdown(' - Enter some text, preferably a lot of it.')
st.sidebar.markdown(' - Choose your Word and Sentence Limit - for different models.')
st.sidebar.markdown(' - Hit Submit!')
st.sidebar.markdown(' - You can see the text summarised by different models and their percentage of the original text.')
st.sidebar.markdown(' - Lastly, view some additional stats about the text as well. ')


text = st.text_area("Enter Some Text (between 1000 to 15000 characters)", max_chars = 15000, height = 300)
count = st.slider("Choose A Word Count Limit", min_value = 30, max_value = 75)
sentence_count = st.slider("Choose A Sentence Limit", min_value = 1, max_value = 10)
submit = st.button("Summarize!")
if submit:
    if len(text) < 1000:
        st.write('Please enter a text in English of minimum 1,000 characters')
    else:  
        summary = summarize(text, word_count=count, ratio=1000/count)
        result = (str(len(summary)) + ' characters' + ' ('"{:.0%}".format(len(summary)/len(text)) + ' of original content)')
        st.markdown('___')
        st.write('TextRank Model based on Word Count')
        st.caption(result)
        st.success(summary) 
        
        my_parser = PlaintextParser.from_string(text,Tokenizer('english'))
        lex_rank_summarizer = LexRankSummarizer()
        lexrank_summary = lex_rank_summarizer(my_parser.document,sentence_count)
        summary = ''
        for sentence in lexrank_summary:
                summary = summary + str(sentence)
        summary = summary.replace(".",". ")
        result = (str(len(summary)) + ' characters' + ' ('"{:.0%}".format(len(summary)/len(text)) + ' of original content)')
        st.markdown('___')
        st.write('LexRank Model based on Sentence Count')
        st.caption(result)
        st.success(summary)

        stopwords = set(stopwords.words("english"))
        words = word_tokenize(text)
        filtered_words = [word.lower() for word in words if word not in stopwords]
        freq_dict = {}
        for word in filtered_words:
            if word not in freq_dict:
                freq_dict[word] = 1
            else:
                freq_dict[word] += 1
        sentences = sent_tokenize(text)
        sentence_score = {}
        for sentence in sentences:
            for word, freq in freq_dict.items():
                if word in sentence.lower():
                    if sentence not in sentence_score:
                        sentence_score[sentence] = freq
                    else:
                        sentence_score[sentence] += freq
        sum = 0
        for sentence in sentence_score:
            sum += sentence_score[sentence]
        avg = int (sum/len(sentence_score))
        summary = ""
        for sentence in sentences:
            if sentence_score[sentence]> 1.2 * avg:
                summary += " " + sentence
        result = (str(len(summary)) + ' characters' + ' ('"{:.0%}".format(len(summary)/len(text)) + ' of original content)')
        st.markdown('___')
        st.write('Scoring Model Based on Word Frequency')
        st.caption(result)
        st.success(summary)

        auto_abstractor = AutoAbstractor()
        auto_abstractor.tokenizable_doc = SimpleTokenizer()
        auto_abstractor.delimiter_list = [".", "\n"]
        abstractable_doc = TopNRankAbstractor()
        result_dict = auto_abstractor.summarize(text, abstractable_doc)
        summary = ""
        sum = 0
        sentences = result_dict['summarize_result']
        scores = result_dict['scoring_data']
        for score in scores:
            sum += score [1]
        avg = sum/len(scores)
        for i in range(len(scores)):
            if scores[i][1] > 1.2 * avg:
                summary += " " + sentences [i]
        result = (str(len(summary)) + ' characters' + ' ('"{:.0%}".format(len(summary)/len(text)) + ' of original content)')
        st.markdown('___')
        st.write('Py Summarizing Model')
        st.caption(result)
        st.success(summary)

        st.markdown('___')
        st.write('Some More About This Text:')
        read_time = textstat.reading_time(text)
        reading_ease = textstat.flesch_reading_ease(text)
        lexical_richness = round(len(set(words))/ len (words), 2)
        sentence_count = textstat.sentence_count(text)
        st.text('Reading Time in ms')
        st.write(read_time)
        st.text('Text Complexity (from 0 - 100) ')
        st.write(reading_ease)
        st.text('Lexical Richness (proportion of unique words)')
        st.write(lexical_richness)
        st.text('Number of sentences')
        st.write(sentence_count)


with st.expander("What I Learned From This Project"):
    st.markdown("I learned the basics of NLP - from tokenisation to stopwords removal, as well as basic text analytics from textstat.")
    st.markdown("I learned about different NLP libraries and the functions they provide - for example, there are so many possible summarizers and they all work differently.")
    st.markdown("I tried out BART and t5 Transformers from Huggingface as well but they did not perform as well, they probably need a much larger chunk of text.")
        




