import re
import numpy as np
from nltk import sent_tokenize, word_tokenize
from nltk.cluster.util import cosine_distance
from nltk.corpus import stopwords
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


# added english stopwords
stop_words = stopwords.words('english')
MULTIPLE_WHITESPACE_PATTERN = re.compile(r"\s+", re.UNICODE)


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

    def analyze(self, text, stop_words=None):
        self.text_str = text
        self.sentences = sent_tokenize(self.text_str)

        # 특수문자 제거
        eng_sentences = []
        for sent in self.sentences:
            sent = re.sub('[^a-zA-Z0-9]', ' ', sent)
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
        # print('simmat\n', similarity_matrix, similarity_matrix.shape)
        # print('cormat\n', correlation_matrix, correlation_matrix.shape)
        print(self.pr_vector)
        print()
        print(self.pr_vector2)


text_str = """
Small-business owners are taking legal action against the government, calling the social distancing rules unfair and saying the restrictions have greatly diminished their chances of economic survival. On Monday, a group representing coffee shop owners in South Korea announced that around 200 of its members would lodge a suit against the government later this week demanding compensation for business lost as a result of the government’ social distancing rules. In the suit, to be filed with the Seoul Central District Court on Thursday, the group said it would demand around 1 billion won ($908,638), or 5 million won for each cafe owner involved. Since late November, when Level 2 social distancing rules were imposed in Seoul, Incheon and Gyeonggi Province, coffee shops in the capital region have been barred from providing dine-in services.
That restriction was extended to coffee shops in other parts of the country in early December, when the government imposed Level 2.5 social distancing rules in the capital region and Level 2 rules for the rest of the country. “We are filing this lawsuit out of desperation as the government’s COVID-19 regulations have put our lives on the edge,” said Ko Jang-soo, head of the cafe owners’ association. “We ask the government to prepare fair and consistent measures.” Cafe owners are not alone in taking issue with the distancing guidelines.
According to a survey of 1,018 small businesses carried out by the Korean Federation of Micro Enterprises in November, 70.8 percent of the respondents said their 2020 sales had dropped from a year earlier, with the average loss estimated at 37.4 percent.
Some 53.5 percent of respondents found the government’s support insufficient, with nearly half of that number calling it a temporary solution.
Since last week, many owners of coffee shops, bars and internet cafes have staged protests against the government. Owners of indoor sports facilities started the movement after the apparent suicide of a gym owner in Daegu who was experiencing financial difficulties.
They argue that their businesses should not face tougher restrictions than restaurants, which are still allowed to offer dining in. The restrictions have drawn consumers away and left the owners with debts piling up, they say.
“I pay around 10 million won in operation fees for the fitness club, and with the COVID-19 pandemic, I am now left with 1.9 million won in my bank account and 90 million won in bank debt,” said Oh Sung-young, head of a gym owners’ association, in a Facebook post Thursday.
Another representative group for owners of indoor fitness centers followed the lead of coffee shop owners to also file a lawsuit of 203 plaintiffs with the Seoul Western District Court on Tuesday, demanding 5 million won for each gym owner involved.
The group had filed a separate lawsuit last month asking for 765 million won from the government for fitness center owners’ businesses losses from social distancing rules.
Businesses argue that the government has not done proper research into the steps small merchants have taken to prevent the spread of the coronavirus within their facilities.
A number of cafes have spaced out their tables to ensure greater distance among customers, and gym owners have installed dividers between exercise machines so that no droplets can pass from one person to one another.
Some business owners have bought extra hand sanitizer and cleaning supplies to ensure safety for customers, only to find that no one was allowed inside.
“I simply wasted my own precious money to buy all these useless plastic dividers that are now covered in dust,” said an independent cafe owner in Gangnam District, southern Seoul, who wished to remain anonymous.
“Nobody is going to pay me back for the dividers, I get that. But the only thing I want from politicians and civil workers is for them to just actually visit the businesses that are in trouble and learn on-site how much effort businesses have made to follow reasonable virus control measures.”
In response, the government and the ruling Democratic Party pledged to come up with additional steps to help small businesses recoup their losses.
“Our hearts are heavy in listening to desperate cries from small-business owners and proprietors in sectors that experienced suspension or limits to their operation,” said Democratic Party Floor Leader Kim Tae-nyeon in a party meeting Monday.
“The Democratic Party and the government will not turn away from the pain of small merchants and concentrate all of our policy efforts to provide support.”
The country on Monday started its third emergency cash relief program, offering up to 3 million won each for around 2.5 million small-business owners, freelancers and contract laborers.
The ruling party is also reviewing a fourth round of disaster relief, to potentially benefit all Koreans, following the lead of party Chairman Lee Nak-yon.
"""

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

