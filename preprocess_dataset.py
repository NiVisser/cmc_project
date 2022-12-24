import pandas as pd
import spacy
import warnings

# nltk.download('vader_lexicon')

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer

warnings.simplefilter(action='ignore', category=FutureWarning)


def extract_relevant_NER(spacy_text):
    """
    Inputs spacy text and counts only relevant NER tags.
    To get an overview of Spacy's NER tags visit:
    https://towardsdatascience.com/named-entity-recognition-with-nltk-and-spacy-8c4a7d88e7da
    Returns NER tags and int variables containing count.
    """
    person_count = 0  # People, including fictional.
    norp_count = 0  # Nationalities or rebellious groups etc.
    organisation_count = 0  # Companies, institutions etc.
    location_count = 0  # Countries, mountains etc.
    event_count = 0  # Wars, sports events etc.
    language_count = 0  # Any named language etc.
    product_count = 0
    ner_tags = []

    for ent in spacy_text:
        ner_tags.append([ent.text, ent.label_])
    if ner_tags:
        for tag in ner_tags:
            if tag[1] == "PERSON":
                norp_count += 1
            elif tag[1] == "NORP":
                person_count += 1
            elif tag[1] == "ORG":
                organisation_count += 1
            elif tag[1] == "GPE" or tag[1] == "LOC":
                location_count += 1
            elif tag[1] == "EVENT":
                event_count += 1
            elif tag[1] == "LANGUAGE":
                language_count += 1
            elif tag[1] == "PRODUCT":
                product_count += 1
    return ner_tags, person_count, norp_count, organisation_count, location_count, event_count, language_count, product_count


def preprocess_dataset():
    """
    Program that calculates total words per comment, total sentences per comment,
    and count of NER tags, based on an input .csv file.
    Return: outputs dataset_processed.csv
    """
    # Total words per comment
    # Sentences per comment
    # NER tags
    nlp = spacy.load("en_core_web_sm")
    sid = SentimentIntensityAnalyzer()

    df = pd.read_csv("dataset_raw.csv", index_col=False)
    df = df.loc[df['comment'] != "[deleted]"]
    df = df.loc[df['comment'] != "[removed]"]
    df = df.rename(columns={df.columns[0]: "id"})
    df['id'] = df['id'].astype('int')
    df['comment_id'] = df['comment_id'].astype('string')

    df = df.sort_values(by=['id'])

    additional_info_df = pd.DataFrame(
        columns=['comment_id', 'sentence_length', 'word_length', 'positive_score', 'negative_score', 'compound_score',
                 'ner_tags', 'person_count', 'norp_count', 'organisation_count', 'location_count',
                 'event_count', 'language_count', 'product_count'])
    for comment_id in df['comment_id']:
        comment_df = df.loc[df['comment_id'] == comment_id]
        comment = comment_df['comment'].iloc[0]

        total_sentences = len(sent_tokenize(comment))
        total_words = len(word_tokenize(comment))
        polarity_score = sid.polarity_scores(comment)
        negative_score, positive_score, compound_score = polarity_score.get('neg'), polarity_score.get(
            'pos'), polarity_score.get('compound')

        spacy_text = nlp(comment)
        ner_tags, person_count, norp_count, organisation_count, location_count, event_count, language_count, product_count = extract_relevant_NER(
            spacy_text.ents)

        new_row = {'comment_id': str(comment_id), 'sentence_length': total_sentences, 'word_length': total_words,
                   'positive_score': positive_score,
                   'negative_score': negative_score, 'compound_score': compound_score, 'ner_tags': ner_tags,
                   'person_count': person_count, 'norp_count': norp_count, 'organisation_count': organisation_count,
                   'location_count': location_count, 'event_count': event_count, 'language_count': language_count,
                   'product_count': product_count}
        additional_info_df = additional_info_df.append(new_row, ignore_index=True)
    final_df = pd.merge(df, additional_info_df, on='comment_id', how='inner')
    final_df = final_df.loc[final_df['word_length'] > 3]
    final_df = final_df.sort_values(by=['created_utc'])
    final_df.to_csv("dataset_processed.csv", index=False)

preprocess_dataset()
