import re
import nltk

nltk.download('stopwords')

from nltk.corpus import stopwords

class LyricsParser:
    def __init__(self, 
                 slurs_file='data/vulgarities/ro_swear_words - slurs-1.csv', 
                 swear_words_file='data/vulgarities/ro_swear_words - vulgarities-1.csv',
                 negative_words='data/sentilex/negative_words_ro.txt',
                 positive_words='data/sentilex/positive_words_ro.txt',
                 chords='data/chords.csv',
                 newline_token=r'\[NL\]'):
        self.slurs_dict = self.load_slurs(slurs_file)
        self.swear_words_dict = self.load_swear_words(swear_words_file)
        
        self.sexual_slurs = self.get_sexual_slurs()
        self.ethnic_slurs = self.get_ethnic_slurs()
        self.swear_words = self.get_swear_words()

        # using https://huggingface.co/datasets/senti-lex/senti_lex/blob/main/data.zip
        self.negative_words = self.load_sentilex_words(negative_words)
        self.positive_words = self.load_sentilex_words(positive_words)

        self.chords = self.load_chords(chords)
        
        self.newline_token = newline_token
        self.ro_stopwords = stopwords.words('romanian')
    
    def load_slurs(self, file_name):
        with open(file_name, 'r') as file:
            line_number = 0
            ngrams = {}
            for line in file:
                if line_number == 0:
                    pass
                else:
                    segments = re.split(r'\,', line)
                    ngram, kind = segments[0], segments[1]
                    ngrams[ngram] = {
                        "id": line_number,
                        "ngram": ngram,
                        "kind": kind[:-1]
                    }
                line_number += 1
            return ngrams

    def load_swear_words(self, file_name):
        with open(file_name, 'r') as file:
            line_number = 0
            ngrams = {}
            for line in file:
                if line_number == 0:
                    pass
                else:
                    ngrams[line[:-1]] = {
                        "id": line_number,
                        "ngram": line[:-1]
                    }
                line_number += 1
            return ngrams

    def load_chords(self, file_name):
        chords = []
        with open(file_name) as file:
            for line in file:
                _chords = line.strip().split(',')
                for chord in _chords:
                    chords.append(chord.strip())
        return chords

    def create_censored_variations(self, ngram):
        variations = []
        for i in range(1, len(ngram)-1):
            copy = list(ngram)
            copy[i] = '*'
            variations.append(''.join(copy))
        
        copy = list(ngram)
        for i in range(1, len(ngram)-1):
            copy[i] = '*'
        
        variations.append(''.join(copy))
        return variations 
    
    def remove_chords(self, x):
        words_cleaned = []
        words = re.split(r"\s", x)
        for word in words:
            if not (word in self.chords):
                words_cleaned.append(word)
        return ' '.join(words_cleaned)
    
    def replace_diacritics(self, text):
        text = text.replace("ă", "a")
        text = text.replace("â", "a")
        text = text.replace("î", "i")
        text = text.replace("ș", "s")
        text = text.replace("ț", "t")
        return text
    
    def load_sentilex_words(self, file_name):
        with open(file_name, 'r') as file:
            line_number = 0
            ngrams = []
            for line in file:
                ngrams.append(line[:-1]) # skip the terminating newline
            return ngrams
    
    def get_sexual_slurs(self):
        slurs = []
        for ngram in self.slurs_dict:
            if self.slurs_dict[ngram]['kind'] == 'sexual':
                slurs.append(self.slurs_dict[ngram]['ngram'])
                ngram_alias = self.replace_diacritics(ngram)
                if (ngram != ngram_alias):
                    slurs.append(ngram_alias)
                censored_variations = self.create_censored_variations(ngram)
                for censored_variation in censored_variations:
                    slurs.append(censored_variation)
        return slurs
    
    def get_ethnic_slurs(self):
        slurs = []
        for ngram in self.slurs_dict:
            if self.slurs_dict[ngram]['kind'] == 'ethnic':
                slurs.append(self.slurs_dict[ngram]['ngram'])
                ngram_alias = self.replace_diacritics(ngram)
                if (ngram != ngram_alias):
                    slurs.append(ngram_alias)
                censored_variations = self.create_censored_variations(ngram)
                for censored_variation in censored_variations:
                    slurs.append(censored_variation)
        return slurs
    
    def get_swear_words(self):
        swear_words = []
        for ngram in self.swear_words_dict:
            swear_words.append(self.swear_words_dict[ngram]['ngram'])
            ngram_alias = self.replace_diacritics(ngram)
            if (ngram != ngram_alias):
                swear_words.append(ngram_alias)
            censored_variations = self.create_censored_variations(ngram)
            for censored_variation in censored_variations:
                swear_words.append(censored_variation)
        return swear_words

    def preprocess_text(self, x, replace_diacritics=True):
        x = str(x)
        x = x.lower()
        
        if replace_diacritics:
            x = self.replace_diacritics(x)
        
        x = re.sub(r'[^a-zA-ZăâîșțĂÂÎȘȚ]', ' ', x)
        ngrams = re.split(r'\s+', x.strip())
        return ngrams
    
    def get_cleaned_verses (self, x):
        x = str(x)
        verses = re.split(self.newline_token, x)
        cleaned_verses = []
        for verse in verses:
            
            cleaned = re.sub(r'[^\w\-\']+', ' ', verse)
            
            # if more than 10% "-" then skip line
            if (len(cleaned) == 0 or cleaned.count('-')/len(cleaned)) >= 0.5:
                continue
            
            cleaned = re.sub(r'-{2,}', ' ', cleaned)
            
            # remove chord symbols
            cleaned = self.remove_chords(cleaned)

            # remove digits
            cleaned = re.sub(r'\d', ' ', cleaned)

            # remove spaces
            cleaned = re.sub(r'\s+', ' ', cleaned)
            cleaned = re.sub(r'^\s|\s$', '', cleaned)
            
            if (len(cleaned) > 0):
                cleaned_verses.append(cleaned)
        return cleaned_verses

    def get_mean_verse_length(self, x):
        verse_count = self.get_verse_count(x)
        if verse_count > 0:
            return self.get_word_count(x) / verse_count
        else:
            return 0.0
    
    def restore_newlines(self, x):
        return re.sub(self.newline_token, "\n", x)

    def get_verse_count(self, x):
        return len(self.get_cleaned_verses(x))
    
    def get_cleaned_phrases(self, x):
        x = str(x)
        x = self.restore_newlines(x)
        phrases = re.split(r'[\.\;\!\?\:]', x)
        cleaned_phrases = []
        for phrase in phrases:
            cleaned = re.sub(r'[^\w\-\']+', ' ', phrase)
            
            # if more than 50% "-" then skip line
            if (len(cleaned) == 0 or cleaned.count('-')/len(cleaned)) >= 0.5:
                continue
            
            cleaned = re.sub(r'-{2,}', ' ', cleaned)
            
            # remove chord symbols
            cleaned = self.remove_chords(cleaned)

            # remove digits
            cleaned = re.sub(r'\d', ' ', cleaned)

            # remove spaces
            cleaned = re.sub(r'\s+', ' ', cleaned)
            cleaned = re.sub(r'^\s|\s$', '', cleaned)
            
            if (len(cleaned) > 0):
                cleaned_phrases.append(cleaned)
        return cleaned_phrases

    def get_phrase_count(self, x):
        return len(self.get_cleaned_phrases(x))

    def get_mean_phrase_length(self, x):
        phrase_count = self.get_phrase_count(x)
        if phrase_count > 0:
            return self.get_word_count(x) / phrase_count
        else:
            return 0.0
            
    def get_char_count(self, x):
        verses = self.get_cleaned_verses(x)
        joined = ' '.join(verses)
        return len(joined)
        
    def get_stopword_count(self, x):
        ngrams = self.preprocess_text(x, replace_diacritics=False)
        count = 0
        for ngram in ngrams:
            if ngram in self.ro_stopwords:
                count += 1
        return count

    def get_stopword_ratio(self, x):
        word_count = self.get_word_count(x)
        if word_count > 0:
            return self.get_stopword_count(x) / word_count
        else:
            return 0.0

    def get_word_count(self, x):
        ngrams = self.preprocess_text(x)
        return len([n for n in ngrams if n])

    def get_mean_word_length(self, x):
        ngrams = self.preprocess_text(x)
        ngrams = [n for n in ngrams if n and n not in self.ro_stopwords]
        return sum(len(n) for n in ngrams) / len(ngrams) if len(ngrams) > 0 else 0.0
    
    def get_vocabulary(self, x, exclude_stopwords=True):
        # don't replace diacritics
        ngrams = self.preprocess_text(x, replace_diacritics=False)
        vocab = {}
        for ngram in ngrams:
            if exclude_stopwords and ngram in self.ro_stopwords:
                pass
            elif ngram in vocab:
                vocab[ngram] += 1
            else:
                vocab[ngram] = 1
        labels = [ngram for ngram in vocab]
        frequencies = [vocab[ngram] for ngram in vocab]

        return labels, frequencies

    def get_vocab_size(self, x):
        vocab, _ = self.get_vocabulary(x)
        return len(vocab)

    def get_enjabement_count(self, x):
        verses = self.get_cleaned_verses(x)
        enjambements = 0
        for i, verse in enumerate(verses[:-1]):  # ignore last line
            if not re.search(r'[.!?]$', verse.strip()):
                enjambements += 1
        return enjambements
    
    def get_swear_word_count(self, x):
        ngrams = self.preprocess_text(x)
        count = 0
        for ngram in ngrams:
            if ngram in self.swear_words:
                count += 1
        return count

    def get_swear_word_ratio(self, x):
        word_count = self.get_word_count(x)
        if word_count > 0:
            return self.get_swear_word_count(x) / word_count
        else:
            return 0.0
    
    def get_ethnic_slur_count(self, x):
        ngrams = self.preprocess_text(x)
        count = 0
        for ngram in ngrams:
            if ngram in self.ethnic_slurs:
                count += 1
        return count

    def get_ethnic_slur_ratio(self, x):
        word_count = self.get_word_count(x)
        if word_count > 0:
            return self.get_ethnic_slur_count(x) / word_count
        else:
            return 0.0
    
    def get_sexual_slur_count(self, x):
        ngrams = self.preprocess_text(x)
        count = 0
        for ngram in ngrams:
            if ngram in self.sexual_slurs:
                count += 1
        return count

    def get_sexual_slur_ratio(self, x):
        word_count = self.get_word_count(x)
        if word_count > 0:
            return self.get_sexual_slur_count(x) / word_count
        else:
            return 0.0
            
    def get_all_vulgarities_count(self, x):
        return self.get_swear_word_count(x) + self.get_ethnic_slur_count(x) + self.get_sexual_slur_count(x)

    def get_all_vulgarities_ratio(self, x):
        word_count = self.get_word_count(x)
        if word_count > 0:
            return self.get_all_vulgarities_count(x) / word_count
        else:
            return 0.0
            
    def get_sentiment_scores(self, x, ratio='all'): # all = sentiment_i_count/word_count, vulg = sentiment_i_count/sentiment_count
        count = positive_count = negative_count = 0
        word_count = self.get_word_count(x)
        
        ngrams = self.preprocess_text(x)
        
        for ngram in ngrams:
            if ngram in self.positive_words:
                positive_count += 1
                count += 1
            elif ngram in self.negative_words:
                negative_count += 1
                count += 1
        
        if ratio == 'all':
            count = word_count
        
        if count > 0:
            return { 'positive_sentiment': positive_count / count, 'negative_sentiment': negative_count / count }
        else:
            return { 'positive_sentiment': 0.0, 'negative_sentiment': 0.0 }

    def get_top_word_by_frequency(self, x):
        vocab, freq = self.get_vocabulary(x)
        
        # invalid vocabulary
        if len(freq) == 0:
            return ''
            
        max_freq = max(freq)

        # not enough repetitions
        if max_freq == 1:
            return ''
            
        argmax_freq = freq.index(max_freq)
        return vocab[argmax_freq]

    def longest_repeated_sequence(self, vector, target):
        max_length = 0
        current_length = 0
        
        for elem in vector:
            if elem == target:
                current_length += 1
                max_length = max(max_length, current_length)
            else:
                current_length = 0
        
        return max_length
            
    def get_repetitions_max_count(self, x):
        top_word = self.get_top_word_by_frequency(x)
        verses = self.get_cleaned_verses(x)
        max_count = 0
        for verse in verses:
            sequence = re.split(r'\s', verse)
            count = self.longest_repeated_sequence(sequence, top_word)
            if count > max_count:
                max_count = count
        return max_count

    def get_repetitions_position(self, x):
        verses = self.get_cleaned_verses(x)
        beginning = 0
        end = 0
        for verse in verses:
            top_word = self.get_top_word_by_frequency(verse)
            if not top_word:
                continue
            ngrams_verse = self.preprocess_text(verse, replace_diacritics=False)
            midpoint = len(ngrams_verse) / 2.0
            pos = ngrams_verse.index(top_word)
            if pos >= midpoint:
                end += ngrams_verse.count(top_word)
            elif pos < midpoint:
                beginning += ngrams_verse.count(top_word)
        return { 'beginning': beginning, 'end': end }
    