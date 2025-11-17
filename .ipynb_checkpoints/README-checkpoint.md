# 2025-ro-song-lyrics-classification-public
2025 Romanian Song Lyrics Classification - Assets, Libraries, Data

```python

parser = LyricsParser(
    newline_token=r'\n',
    negative_words='data/sentilex_v2/negative_words_ro.txt',
    positive_words='data/sentilex_v2/positive_words_ro.txt')
```

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

df['negative_sentiment_sentilex_v2'] = df['sentiment_scores'].progress_apply(lambda x: x['negative_sentiment'])
df['positive_sentiment_sentilex_v2'] = df['sentiment_scores'].progress_apply(lambda x: x['positive_sentiment'])
df['repetitions_beginning'] = df['repetitions_position'].progress_apply(lambda x: x['beginning'])
df['repetitions_end'] = df['repetitions_position'].progress_apply(lambda x: x['end'])
```