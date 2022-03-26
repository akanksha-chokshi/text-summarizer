# text-summarizer
In this project, I implement **Extractive Text Summarisation** (selecting sentences from within the entered chunk of text that best encapsulate the content of the entire text, compared to **Abstractive Text Summarisation** which generates original text that summarised the block of text).

Extractive Text Summarisation involves examining each sentence within the chunk of text and understanding how it relates to other sentences within the text chunk - how relevant and important it is in comparison. 

To implement this, I used four different models - the TextRank model, the LexRank model, a simple similarity score based model and PySummarization. Each of these uses a different approach to determine which sentences are the most relevant and important.

**TextRank** and **LexRank** are both graph-based ranking algorithms, inspired by Google’s PageRank. 

What **PageRank** does is that it converts each website into a node in a graph and connects websites that link to each other via edges. It ranks each site based on how well-connected it is (number of sites that link to it) but also how well-connected it is to other well-ranked sites (quality of sites that link to it). PageRank is used to rank websites in Google Search. 

**Check out my implementation of the PageRank algorithm [here](https://github.com/akanksha-chokshi/pagerank).**

**TextRank** and **LexRank** follow a similar structure: they first extract keywords from the chunk of text and turn each keyword into a node. Keywords within a certain “cooccurrence range” (the average sentence length) are linked to each other via edges. The sentences that are linked to the most sentences (which are also linked to the most sentences) qualify as summarising sentences.

Where the two algorithms differ is how they calculate the **“similarity score”** between two sentences. TextRank uses the number of words that overlap between two sentences divided by their normalised length as the similarity metric. LexRank uses cosine similarity between sentences vectorised by tf-IDF (numerical representation of words based on their frequency of occurrence in a particular document).

**In my own implementation of text summarization**, I first preprocess the text using stopwords removal, tokenise each word and build a frequency dictionary for each word. I then “score” each sentence based on the frequency score of the words it contains. I then find the average of sentence scores and only select the sentences that are above 1.2 times the average.

The last technique I use is **PySummarization**, a Python library that performs text summarization using an LSTM and sequence-to-sequence learning techniques.

I also tried BART and t5 transformers from HuggingFace but they did not work very well with a text chunk of this size. I assume it’s because they would need a much larger text chunk to be effective.
