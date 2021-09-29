import lyricsgenius as genius
import pandas as pd
import string

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('wordnet')


def search_data(query, n, access_token):
    """
    This function uses the library lyricsgenius to extract the fields
    title, artist, album, date and lyrics and stores them into a pandas dataframe
    parameters:
    query = artist or band to search
    n = max numbers of songs
    access_token = your access token of the genius api
    """

    api = genius.Genius(access_token)

    list_lyrics = []
    list_title = []
    list_artist = []
    #list_album = []
    #list_year = []

    artist = api.search_artist(query, max_songs=n, sort='popularity')
    songs = artist.songs
    for song in songs:
        list_lyrics.append(song.lyrics)
        list_title.append(song.title)
        list_artist.append(song.artist)
        #list_album.append(song.album)
        #list_year.append(song.year)

    df = pd.DataFrame({'artist': list_artist, 'title': list_title, 'lyric': list_lyrics})

    return df


def clean_lyrics(df, column):
    """
    This function cleans the words without importance and fix the format of the  dataframe's column lyrics
    parameters:
    df = dataframe
    column = name of the column to clean
    """
    df = df
    df[column] = df[column].str.lower()
    df[column] = df[column].str.replace(r"verse |[1|2|3]|chorus|bridge|outro", "").str.replace("[", "").str.replace("]",
                                                                                                                    "")
    df[column] = df[column].str.lower().str.replace(r"instrumental|intro|guitar|solo", "")
    df[column] = df[column].str.replace("\n", " ").str.replace(r"[^\w\d'\s]+", "").str.replace("efil ym fo flah", "")
    df[column] = df[column].str.strip()

    return df


def lyrics_to_words(document):
    """
    This function splits the text of lyrics to  single words, removing stopwords and doing the lemmatization to each word
    parameters:
    document: text to split to single words
    """
    stop_words = set(stopwords.words('english'))
    exclude = set(string.punctuation)
    lemma = WordNetLemmatizer()
    stopwordremoval = " ".join([i for i in document.lower().split() if i not in stop_words])
    punctuationremoval = ''.join(ch for ch in stopwordremoval if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punctuationremoval.split())
    return normalized
