# Improving Sentiment Analysis with Multi-task Learning of Negation

Jeremy Barnes [jeremycb@ifi.uio.no]
Lilja Øvrelid [liljao@ifi.uio.no]
Erik Velldal [erikve@ifi.uio.no]

Sentiment analysis is directly affected by _compositional phenomena_ in language that act on the prior polarity of the words and phrases found in the text. __Negation__ is the most prevalent of these phenomena and in order to correctly predict sentiment, a classifier must be able to identify negation and disentangle the effect that its scope has on the final polarity of a text. This paper proposes a multi-task approach to explicitly incorporate information about negation in sentiment analysis, which we show outperforms learning negation implicitly in a data-driven manner. We describe our approach, a __cascading neural architecture with selective sharing of LSTM layers__, and show that explicitly training the model with negation as an auxiliary task helps improve the main task of sentiment analysis. The effect is demonstrated across several different standard English-language datasets for both tasks and we analyze several aspects of our system related to its performance, varying types and amounts of input data and different multi-task setups.

## Models
1. Single-task model
2. Multi-task SFU
3. Multi-task CD
4. Transfer-learning

## Datasets
1. SST
2. SemEval
3. SFU
4. Conan Doyle Neg (\*Sem 2012)
5. Streusle

### Requirements

1. Python 3
2. sklearn  ```pip install -U scikit-learn```
3. Pytorch ```pip install torch torchvision```
4. nltk ```pip install nltk```


