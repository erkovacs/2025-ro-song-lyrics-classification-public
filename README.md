# 2025-ro-song-lyrics-classification-public
2025 Romanian Song Lyrics Classification - Assets, Libraries, Data

## Context

This is the repository associated with our article "Music Genre classification using prosodic, stylistic, syntactic, and sentiment-based features". 

## File index
|File||
|---|---|
|```./sentilex```| contains the basic files from which we started, identical to the current  release of SentiLex|
|```./sentilex_v2``` | contains the modified versions of the SentiLex files, as described in the article|
|```./vulgarities``` | lexicon of vulgar words in Romanian|
|```20251117-songs-data.csv``` | the training dataset scraped as described in the article. Contains both the raw lyrics data and two cleaned versions, one using our own approach and one cleaned by an LLM|
|```categories.csv``` | the index of the category pages from tabulaturi.ro|
|```chords.csv``` | a lexicon of all chord symbols, we used it in the cleaning process|
|```genre_mappings.csv```, ```mappings.md``` | the mappings describing the method we used to reduce the number of genres to a manageable amount|
|```lyrics_parser_v2.py```| the code producing the features |
|```sentiment_test.csv``` | the test dataset we used to evaluate our improvements to SentiLex [1]|
|```songs_categories.csv``` | an index of the song and the category|

## Parser usage

The parser is a basic python class. In order to choose the lexicons, we need to provide the information in the constructor:

```python

parser = LyricsParser(
    newline_token=r'\n',
    negative_words='data/sentilex_v2/negative_words_ro.txt',
    positive_words='data/sentilex_v2/positive_words_ro.txt')
```
Then it is possible to apply all the functions to parse the text into the numeric features. Here I am using a Pandas dataframe and the tqdm library to encode the features, mapping them onto new columns on the same dataframe object:

```python
from tqdm import tqdm

tqdm.pandas()

df['ethnic_slur_ratio'] = df['cleaned_text_chatgpt'].progress_apply(parser.get_ethnic_slur_ratio)
df['sexual_slur_ratio'] = df['cleaned_text_chatgpt'].progress_apply(parser.get_sexual_slur_ratio)
df['all_vulgarities_ratio'] = df['cleaned_text_chatgpt'].progress_apply(parser.get_all_vulgarities_ratio)
df['sentiment_scores'] = df['cleaned_text_chatgpt'].progress_apply(parser.get_sentiment_scores)
df['repetitions_max_count'] = df['cleaned_text_chatgpt'].progress_apply(parser.get_repetitions_max_count)
df['repetitions_position'] = df['cleaned_text_chatgpt'].progress_apply(parser.get_repetitions_position)
df['mean_verse_length'] = df['cleaned_text_chatgpt'].progress_apply(parser.get_mean_verse_length)
df['mean_phrase_length'] = df['cleaned_text_chatgpt'].progress_apply(parser.get_mean_phrase_length)
df['char_count'] = df['cleaned_text_chatgpt'].progress_apply(parser.get_char_count)
df['stopword_ratio'] = df['cleaned_text_chatgpt'].progress_apply(parser.get_stopword_ratio)
df['word_count'] = df['cleaned_text_chatgpt'].progress_apply(parser.get_word_count)
df['mean_word_length'] = df['cleaned_text_chatgpt'].progress_apply(parser.get_mean_word_length)
df['vocab_size'] = df['cleaned_text_chatgpt'].progress_apply(parser.get_vocab_size)
df['enjabement_count'] = df['cleaned_text_chatgpt'].progress_apply(parser.get_enjabement_count)
df['swear_word_ratio'] = df['cleaned_text_chatgpt'].progress_apply(parser.get_swear_word_ratio)

df['sentiment_scores_sentilex'] = df['cleaned_text_chatgpt'].progress_apply(parser.get_sentiment_scores)

# To split the negative and positive sentiment into separate features (normally they are combined),
# we need to adjust them a bit (get only the nefative and positive components respectively)
# Same goes for the positions of the repetitions, we need to parse the end and beginning repetitions
# into their own features

df['negative_sentiment_sentilex_v2'] = df['sentiment_scores'].progress_apply(lambda x: x['negative_sentiment'])
df['positive_sentiment_sentilex_v2'] = df['sentiment_scores'].progress_apply(lambda x: x['positive_sentiment'])
df['repetitions_beginning'] = df['repetitions_position'].progress_apply(lambda x: x['beginning'])
df['repetitions_end'] = df['repetitions_position'].progress_apply(lambda x: x['end'])
```

## References

[1] - Dumitrescu et al. (2020), The birth of Romanian BERT, Findings of the Association for Computational Linguistics: EMNLP 2020, https://aclanthology.org/2020.findings-emnlp.387/

## Cite us

-- TO BE ADDED -- 
