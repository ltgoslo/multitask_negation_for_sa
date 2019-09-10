# Improving Sentiment Analysis with Multi-task Learning of Negation

[Jeremy Barnes](jeremycb@ifi.uio.no)
[Lilja Øvrelid](liljao@ifi.uio.no)
[Erik Velldal](erikve@ifi.uio.no)

Sentiment analysis is directly affected by _compositional phenomena_ in language that act on the prior polarity of the words and phrases found in the text. __Negation__ is the most prevalent of these phenomena and in order to correctly predict sentiment, a classifier must be able to identify negation and disentangle the effect that its scope has on the final polarity of a text. This paper proposes a multi-task approach to explicitly incorporate information about negation in sentiment analysis, which we show outperforms learning negation implicitly in a data-driven manner. We describe our approach, a __cascading neural architecture with selective sharing of LSTM layers__, and show that explicitly training the model with negation as an auxiliary task helps improve the main task of sentiment analysis. The effect is demonstrated across several different standard English-language datasets for both tasks and we analyze several aspects of our system related to its performance, varying types and amounts of input data and different multi-task setups.

## Models
1. Single-task model
2. Multi-task SFU
3. Multi-task CD
4. Transfer-learning

## Datasets
1. [SST](https://nlp.stanford.edu/sentiment/treebank.html)
2. [SemEval 2013 SA task](https://www.cs.york.ac.uk/semeval-2013/task2/)
3. [SFU Review Corpus](https://www.sfu.ca/~mtaboada/SFU_Review_Corpus.html)
4. [Conan Doyle Neg (\*Sem 2012)](https://www.clips.uantwerpen.be/sem2012-st-neg/)
5. [Streusle Dataset](https://github.com/nert-nlp/streusle)

## Embeddings
You can find the embeddings used in the experiments [here](https://drive.google.com/open?id=1GpyF2h0j8K5TKT7y7Aj0OyPgpFc8pMNS). Untar the file and use the 'google.txt' embeddings.

## Requirements

1. Python 3
2. sklearn  ```pip install -U scikit-learn```
3. Pytorch ```pip install torch torchvision torchtext```
4. nltk ```pip install nltk```
5. matplotlib ```pip install matplotlib```
6. tqdm ```pip install tqdm```

