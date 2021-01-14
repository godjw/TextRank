import re
import numpy as np
import nltk
from nltk import sent_tokenize, word_tokenize
from nltk.cluster.util import cosine_distance
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

# added english stopwords
stop_words = stopwords.words('english')
MULTIPLE_WHITESPACE_PATTERN = re.compile(r"\s+", re.UNICODE)
lemmatizer = WordNetLemmatizer()

def normalize_whitespace(text):
    """
    Translates multiple whitespace into single space character.
    If there is at least one new line character chunk is replaced
    by single LF (Unix new line) character.
    """
    return MULTIPLE_WHITESPACE_PATTERN.sub(_replace_whitespace, text)


def _replace_whitespace(match):
    text = match.group()

    if "\n" in text or "\r" in text or "\r\n" in text:
        return "\n"
    else:
        return " "


def is_blank(string):
    """
    Returns `True` if string contains only white-space characters
    or is empty. Otherwise `False` is returned.
    """
    return not string or string.isspace()


def get_symmetric_matrix(matrix):
    """
    Get Symmetric matrix
    :param matrix:
    :return: matrix
    """
    return matrix + matrix.T - np.diag(matrix.diagonal())


def core_cosine_similarity(vector1, vector2):
    """
    measure cosine similarity between two vectors
    :param vector1:
    :param vector2:
    :return: 0 < cosine similarity value < 1
    """
    return 1 - cosine_distance(vector1, vector2)


def pos_tagger(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def lemmatize_text(text):
    lemmatized_text = ""
    for sentence in text:
        # tokenize the sentence and find the POS tag for each token
        pos_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))

        # we use our own pos_tagger function to make things simpler to understand.
        wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))

        lemmatized_sentence = []
        for word, tag in wordnet_tagged:
            if tag is None:
                # if there is no available tag, append the token as is
                lemmatized_sentence.append(word)
            else:
                # else use the tag to lemmatize the token
                lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
        lemmatized_sentence = " ".join(lemmatized_sentence)
        lemmatized_text += lemmatized_sentence + " "
    return lemmatized_text

class TextRank4Sentences():
    def __init__(self):
        self.damping = 0.85  # damping coefficient, usually is .85
        self.min_diff = 1e-5  # convergence threshold
        self.steps = 100  # iteration steps
        self.text_str = None
        self.sentences = None
        self.pr_vector = None

        # added for tf-idf
        self.pr_vector2 = None
        self.tfidf = TfidfVectorizer()

    def _sentence_similarity(self, sent1, sent2, stopwords=None):
        if stopwords is None:
            stopwords = []

        sent1 = [w.lower() for w in sent1]
        sent2 = [w.lower() for w in sent2]

        all_words = list(set(sent1 + sent2))

        vector1 = [0] * len(all_words)
        vector2 = [0] * len(all_words)

        # build the vector for the first sentence
        for w in sent1:
            if w in stopwords:
                continue
            vector1[all_words.index(w)] += 1

        # build the vector for the second sentence
        for w in sent2:
            if w in stopwords:
                continue
            vector2[all_words.index(w)] += 1

        return core_cosine_similarity(vector1, vector2)

    def _build_sent_graph(self, sentences):
        tfidf_mat = self.tfidf.fit_transform(sentences).toarray()
        graph_sentence = np.dot(tfidf_mat, tfidf_mat.T)
        sm = np.zeros([len(sentences), len(sentences)])

        for idx1 in range(len(graph_sentence)):
            for idx2 in range(len(graph_sentence)):
                if idx1 == idx2:
                    continue
                sm[idx1][idx2] = core_cosine_similarity(graph_sentence[idx1], graph_sentence[idx2])

        sm = get_symmetric_matrix(sm)

        # Normalize matrix by column
        norm = np.sum(sm, axis=0)
        sm_norm = np.divide(sm, norm, where=norm != 0)  # this is to ignore the 0 element in norm

        return sm_norm
    
    def _build_similarity_matrix(self, sentences, stopwords=None):
        # create an empty similarity matrix
        sm = np.zeros([len(sentences), len(sentences)])

        for idx1 in range(len(sentences)):
            for idx2 in range(len(sentences)):
                if idx1 == idx2:
                    continue

                sm[idx1][idx2] = self._sentence_similarity(sentences[idx1], sentences[idx2], stopwords=stopwords)

        # Get Symmeric matrix
        sm = get_symmetric_matrix(sm)

        # Normalize matrix by column
        norm = np.sum(sm, axis=0)
        sm_norm = np.divide(sm, norm, where=norm != 0)  # this is to ignore the 0 element in norm

        return sm_norm

    def _run_page_rank(self, similarity_matrix):

        pr_vector = np.array([1] * len(similarity_matrix))

        # Iteration
        previous_pr = 0
        for epoch in range(self.steps):
            pr_vector = (1 - self.damping) + self.damping * np.matmul(similarity_matrix, pr_vector)
            if abs(previous_pr - sum(pr_vector)) < self.min_diff:
                break
            else:
                previous_pr = sum(pr_vector)

        return pr_vector

    def _get_sentence(self, index):

        try:
            return self.sentences[index]
        except IndexError:
            return ""

    def get_top_sentences(self, number=5):

        top_sentences = {}

        if self.pr_vector is not None:

            sorted_pr = np.argsort(self.pr_vector)
            sorted_pr = list(sorted_pr)
            sorted_pr.reverse()

            index = 0
            for epoch in range(number):
                # print(str(sorted_pr[index]) + " : " + str(self.pr_vector[sorted_pr[index]]))
                sent = self.sentences[sorted_pr[index]]
                sent = normalize_whitespace(sent)
                top_sentences[index] = sent + " : " + str(self.pr_vector[sorted_pr[index]])
                index += 1

        return top_sentences

    def get_top_sentences2(self, number=5):

        top_sentences = {}

        if self.pr_vector2 is not None:

            sorted_pr = np.argsort(self.pr_vector2)
            sorted_pr = list(sorted_pr)
            sorted_pr.reverse()

            index = 0
            for epoch in range(number):
                # print(str(sorted_pr[index]) + " : " + str(self.pr_vector[sorted_pr[index]]))
                sent = self.sentences[sorted_pr[index]]
                sent = normalize_whitespace(sent)
                top_sentences[index] = sent + " : " + str(self.pr_vector2[sorted_pr[index]])
                index += 1

        return top_sentences

    def analyze(self, text, stop_words=None, lem_flag=0):
        self.text_str = text
        self.sentences = sent_tokenize(self.text_str)

        if lem_flag == 0:
            temp = self.sentences
        else:
            temp = sent_tokenize(lemmatize_text(self.sentences))
        # 특수문자 제거
        eng_sentences = []
        for sent in temp:
            sent = re.sub('[^a-zA-Z]', ' ', sent)
            eng_sentences.append(sent)
        tokenized_sentences = [word_tokenize(sent) for sent in eng_sentences]

        # stopwords를 제거한 tokenized sentences
        rm_tokenized_sentences = []
        for sent in tokenized_sentences:
            temp = []
            for word in sent:
                if word not in stop_words:
                    temp.append(word)
            rm_tokenized_sentences.append(' '.join(temp))

        correlation_matrix = self._build_sent_graph(rm_tokenized_sentences)
        similarity_matrix = self._build_similarity_matrix(tokenized_sentences, stop_words)

        self.pr_vector = self._run_page_rank(similarity_matrix)
        self.pr_vector2 = self._run_page_rank(correlation_matrix)

        print(self.pr_vector)
        print()
        print(self.pr_vector2)


text_str = """The coronavirus pandemic and a series of hedge fund turmoils last year have impeded the growth of private funds managed by asset management companies in South Korea, data showed Thursday.

According to data by the Korea Financial Investment Association, the combined assets managed by Korea-domiciled private funds grew 5.7 percent to 438.7 trillion won ($398.4 billion) in 2020, far lower than their 2019 growth which stood at 23.5 percent. Between 2014 and 2019, the funds grew 18.8 percent annually. Private funds refer to instruments composed of no more than 49 end-investors under Korean rules.

The size of private funds dedicated to outbound investment increased 12.6 percent in 2020, slower than it had been over the past five years, when it grew 26.1 percent each year on average.

The trend is largely attributable to mounting public distrust in private funds that are subject to looser regulations and restrictions on due diligence of foreign real assets -- considered the favored destinations of such funds -- wrote Oh Gwang-young, an analyst at Shinyoung Securities.

“The shady performances by a few private funds dedicated to alternative assets have sagged investor sentiment, while COVID-19 hampered the creation of new investments (in private funds),” Oh said.

Korea in 2020 suffered trillion-won misselling scandals involving Korean hedge funds Lime Asset Management and Optimus Asset Management -- accused of misrepresenting fund performances to end-investors -- as well as commercial banks‘ misselling of securities linked to derivatives tied to German bond yields. This led to a 5.7 percent fall of hedge fund assets under management by private funds in 2020, for the first time in the record since 2015.

The growth of private funds’ investment in real estate and infrastructure -- which account for nearly 50 percent of the total assets under management -- has also slowed down to nearly half of the level of the five-year average.

This is associated with the decrease in outbound real asset investment of Korean investors. According to intelligence firm Real Capital Analytics, Korean investors‘ cross-border investments came to a total of $8.5 billion in 2020, shrinking to less than half of $18.4 billion in 2019.

Moreover, another set of data indicated that Korean investments in foreign alternative assets are still laden with risks. According to data from the watchdog Financial Supervisory Service, nearly 16 percent of outbound real asset investment by investment banks -- which connect sellers with institutional investors by retaining the sellers’ assets and later selling it down to the institutions -- are categorized as being either prone to delinquencies or losses.

Oh said the pace of growth in private funds investing in real assets will pick up the pieces as long as the world resolves COVID-19 issues. But the rebound of hedge funds is unlikely even in a post-COVID-19 era, Oh added."""


tr4sh = TextRank4Sentences()
print('<PageRank Vector>')
tr4sh.analyze(text_str, stop_words)
top_sentence = tr4sh.get_top_sentences(5)
top_sentence_tfidf = tr4sh.get_top_sentences2(5)
print()

print("<Summarization Using Cosine Similarity>")
for k, v in top_sentence.items():
    print("%d. %s"%(k + 1, v))
print()

print("<Summarization Using TF-IDF & Cosine Similarity>")
for k, v in top_sentence_tfidf.items():
    print("%d. %s"%(k + 1, v))
print()

print('<PageRank Vector(lemmatized)>')
tr4sh.analyze(text_str, stop_words, lem_flag=1)
top_sentence = tr4sh.get_top_sentences(5)
top_sentence_tfidf = tr4sh.get_top_sentences2(5)
print()

print("<Summarization Using Cosine Similarity + Lemmatization>")
for k, v in top_sentence.items():
    print("%d. %s"%(k + 1, v))
print()

print("<Summarization Using TF-IDF & Cosine Similarity + Lemmatization>")
for k, v in top_sentence_tfidf.items():
    print("%d. %s"%(k + 1, v))







