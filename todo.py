#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[4]:


# -*- coding: utf-8 -*-
from flask import request, jsonify
from flask_restx import Resource, Api, Namespace

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.cluster.util import cosine_distance
import kss
import numpy as np
from konlpy.tag import Mecab
from collections import OrderedDict
import math
import re

mecab = Mecab()
def cosine_similarity(sent1, sent2):
    # 입력으로 들어온 두 문장 벡터간의 코사인 유사도를 구하는 함수
    tmp = math.sqrt(np.dot(sent1, sent1)) * math.sqrt(np.dot(sent2, sent2))
    if tmp == 0:
        return 0
    else:
        return np.dot(sent1, sent2) / tmp


class SentenceTokenizer(object):
    def __init__(self):
        # 한국어 형태소 분석기로는 Mecab 사용
        self.mecab = Mecab()


    def text2sentence(self, text):
        # 한국어 문장단위 토큰화
        sentences = kss.split_sentences(text)
        return sentences

    def list2sentence(self, text):
        sent = []
        for sentence in text:
            sent.append(' '.join(sentence))
        return sent

    def get_nouns(self, sentences):
        # 텍스트에서 명사만 추출
        nouns = []
        for sentence in sentences:
            if sentence != '':
                nouns.append(' '.join([noun for noun in self.mecab.nouns(str(sentence)) if len(noun) > 1]))
        return nouns

    def get_numbers(self, sentences):
        numbers = []
        for sentence in sentences:
            temp = sentence.split()
            for num, pos in self.mecab.pos(str(sentence)):
                if pos == 'SN':
                    for word in temp:
                        if num in word:
                            numbers.append(word)
        return numbers


class GraphMatrix(object):
    def __init__(self):
        self.tfidf = TfidfVectorizer()
        self.mecab = Mecab()
        self.sentence_graph = []

    def make_sentence_graph(self, sentences):
        self.sentence_graph = np.zeros([len(sentences), len(sentences)])
        tfidf_mat = self.tfidf.fit_transform(sentences).toarray()
        for i in range(len(sentences)):
            for j in range(len(sentences)):
                if i == j:
                    continue
                self.sentence_graph[i][j] = cosine_similarity(tfidf_mat[i], tfidf_mat[j])

        self.sentence_graph = self.sentence_graph + self.sentence_graph.T - np.diag(self.sentence_graph.diagonal())
        norm = np.sum(self.sentence_graph, axis=0)
        self.sentence_graph = np.divide(self.sentence_graph, norm, where=norm != 0)
        return self.sentence_graph


class Rank(object):
    def __init__(self):
        self.damping = 0.85  # damping factor
        self.min_diff = 1e-5  # convergence thershold
        self.steps = 100  # iteration steps
        self.tr_vector = None

    def get_rank(self, graph):
        tr_vector = np.array([1] * len(graph))
        previous_tr = 0
        for epoch in range(self.steps):
            tr_vector = (1 - self.damping) + self.damping * np.matmul(graph, tr_vector)
            if abs(previous_tr - sum(tr_vector)) < self.min_diff:
                break
            else:
                previous_tr = sum(tr_vector)

        return tr_vector


class TextRank(object):
    def __init__(self, text):
        self.sent_tokenize = SentenceTokenizer()
        self.tokenized_sentence = self.sent_tokenize.text2sentence(text)  # 문장 단위로 토큰화 된 텍스트
        self.sent_num = len(self.tokenized_sentence)  # 문장의 개수를 저장하는 변수 (요약의 비율을 정할 때 이용)
        self.nouns = self.sent_tokenize.get_nouns(self.tokenized_sentence)  # 명사 저장
        self.numbers = self.sent_tokenize.get_numbers(self.tokenized_sentence)
        self.graph_matrix = GraphMatrix()

        self.mecab = Mecab()
        self.pos = ['NNG', 'NNP', 'VA', 'SL']

        # 문장 벡터를 명사, 형용사, 영어를 이용하여 생성
        self.filtered_sentence = self.filter_sentences(self.tokenized_sentence)
        # tf-idf vectorizer를 사용하기 위해, 단어들의 list로 구성되어 있는 list를 string으로 구성되어 있는 list로 변
        self.f_sent = self.sent_tokenize.list2sentence(self.filtered_sentence)

        # 어떤 방식으로 문장 벡터를 생성할지 비교
        #self.sent_graph = self.graph_matrix.make_sentence_graph(self.f_sent)
        self.sent_graph = self.graph_matrix.make_sentence_graph(self.tokenized_sentence)
        #self.sent_graph = self.graph_matrix.make_sentence_graph(self.nouns)

        self.rank = Rank()
        self.sent_rank = self.rank.get_rank(self.sent_graph)

        self.mecab = Mecab()
        self.pos = ['NNG', 'NNP', 'VA', 'SL', 'VV']

        self.word_weight = None
        self.top_sentences = None
        self.keywords = None

        self.summary = None


    def filter_sentences(self, sentences):
        filtered_sent = []
        for sent in sentences:
            temp = []
            for word in self.mecab.pos(sent):
                if word[1] in self.pos:
                    temp.append(word[0])
            filtered_sent.append(temp)

        return filtered_sent

    def get_vocab(self, sentences):
        vocab = OrderedDict()
        i = 0
        for sentence in sentences:
            for word in sentence:
                if word not in vocab:
                    vocab[word] = i
                    i += 1
        return vocab

    def get_token_pairs(self, window_size, sentences):
        token_pairs = []
        for sentence in sentences:
            for i, word in enumerate(sentence):
                for j in range(i + 1, i + window_size + 1):
                    if j >= len(sentence):
                        break
                    pair = (word, sentence[j])
                    if pair not in token_pairs:
                        token_pairs.append(pair)
        return token_pairs

    def symmetrize(self, g):
        return g + g.T - np.diag(g.diagonal())

    def get_word_matrix(self, vocab, token_pairs):
        vocab_size = len(vocab)
        graph = np.zeros((vocab_size, vocab_size), dtype='float')
        for word1, word2 in token_pairs:
            i, j = vocab[word1], vocab[word2]
            graph[i][j] += 1

        graph = self.symmetrize(graph)
        norm = np.sum(graph, axis=0)
        g_norm = np.divide(graph, norm, where=norm != 0)

        return g_norm

    def get_keywords(self, number=12):
        node_weight = OrderedDict(sorted(self.word_weight.items(), key=lambda t: t[1], reverse=True))
        keywords = []
        for i, (key, value) in enumerate(node_weight.items()):
            keywords.append(key)
            if i > number:
                break

        return keywords

    def get_top_sentences(self, number=1):
        top_sentences = {}
        self.summary = []

        if self.sent_rank is not None:
            sorted_tr = np.argsort(self.sent_rank)
            sorted_tr = list(sorted_tr)
            sorted_tr.reverse()

            idx = 0
            for epoch in range(number):
                sent = self.tokenized_sentence[sorted_tr[idx]]
                top_sentences[idx] = sent
                self.summary.append(sent)
                idx += 1

        return top_sentences

    def analyze(self, number=1, window_size=2):
        vocab_list = self.get_vocab(self.filtered_sentence)

        token_pairs = self.get_token_pairs(window_size, self.filtered_sentence)

        graph = self.get_word_matrix(vocab_list, token_pairs)
        word_num = int(len(graph) / 3)
        wr_vector = self.rank.get_rank(graph)

        node_weight = dict()

        for word, index in vocab_list.items():
            node_weight[word] = wr_vector[index]

        self.top_sentences = self.get_top_sentences(number)
        self.word_weight = node_weight
        self.keywords = self.get_keywords(word_num)


def question_generation(summary, ans_candidate):
    question_list = {}
    for sentence in summary:
        sentence = sentence.split()
        question = []
        answer = []
        i = 1  # 빈칸의 총 개수
        d = 0
        for word in sentence:
            match = 0
            for ans in ans_candidate:
                if ans in word:
                    if len(ans) == 1 and len(mecab.pos(word)) == 1:
                        break
                    match = 1
                    break
            if match == 1 and i < 4:
                if ans in answer:
                    idx1 = word.find(ans)
                    idx2 = len(ans)
                    blank_word = word[:idx1] + "[   " + str(answer.index(ans) + 1) + "   ]" + word[idx1+idx2:]
                    question.append(blank_word)
                else:
                    d += 1
                    idx1 = word.find(ans)
                    idx2 = len(ans)
                    blank_word = word[:idx1] + "[   " + str(d) + "   ]" + word[idx1+idx2:]
                    answer.append(ans)
                    question.append(blank_word)
                i = i + 1
            else:
                question.append(word)
        question_list[(' '.join(question))] = answer
    return question_list


# text = input("키워드 추출할 텍스트를 입력하세요\n")
text = """
정의
주의력 결핍 과잉행동장애(ADHD)는 산만함, 과잉행동, 충동성을 특징으로 하는 질환입니다. 이는 12세 이전 발병하고 만성 경과를 보이며, 여러 기능 영역에 지장을 초래합니다. 이 질환 환자 중에는 도덕적인 자제력 부족이나 반항심, 이기심으로 오해받아 괴로워하는 경우가 많습니다. 대략 3~4:1 정도로 남성에서 흔하게 발생합니다. 초등학생 중 13% 정도, 중고등학생 중 7% 정도가 이 질환을 지니고 있습니다. 성인기에 존재하는 산만함이나 충동성에 대해 별개의 시기에 발현한 성인 ADHD로 진단할 것인가, 이전 시기에 발현한 ADHD의 잔재 증상으로 이해할 것인가, 아니면 전혀 다른 별개의 질환에 의한 증상이 집중력 장애의 형태로 나타난 것인가에 대해서는 현재까지도 활발하게 논의하고 있습니다.
원인
ADHD는 뇌 안에서 주의집중 능력을 조절하는 신경전달 물질(도파민, 노르에피네프린 등)이 불균형하여 발생합니다. 주의집중력과 행동을 통제하는 뇌 부위의 구조 및 기능 변화가 ADHD의 발생과 관련이 있습니다. 기타 원인으로는 뇌 손상, 뇌의 후천적 질병, 미숙아 등이 있습니다. 
 
소아 ADHD의 유병률은 일반 인구의 6~9%입니다. 이 중 60~80%는 청소년기까지 계속됩니다. 50%, 즉 소아 ADHD 환자 2명 중 1명은 성인이 되어도 ADHD의 주요 증세나 전체 진단 기준을 충족시키는 증세를 유지합니다. 
증상
ADHD 증상은 환자의 연령대가 증가함에 따라 변화합니다. 과다 활동은 초기 청소년기로 접어들면서 유의미하게 감소합니다. 일부 환자는 충동성, 심한 감정 기복, 주의집중력에서 지속적인 결함을 보입니다. 이를 포함한 성인 ADHD의 증상은 다음과 같습니다. 
① 집중과 집중 유지의 어려움
- 아주 간단한 일임에도 일을 끝마치기 위해 고군분투합니다.
- 일을 끝마치지 못합니다.
- 세밀한 부분을 간과하는 실수가 잦습니다.
- 별로 상관없는 광경이나 소리 때문에 쉽게 산만해집니다.
- 한 가지 일을 하다가 어느새 다른 일을 하고 있습니다.
 
②  과도한 집중
- 책, TV, 컴퓨터 등 흥분과 보상이 있는 일에는 몰입합니다.
- 과도한 집중으로 인해 다른 중요한 일과 시간 개념을 잊어버립니다.
 
③ 비조직화와 건망증
- 정리 정돈을 잘하지 못합니다. 방, 책상, 차가 매우 어지럽습니다.
- 일의 예상 소요시간을 과소평가하는 경향이 있습니다.
- 만성적으로 지각합니다.
- 우선순위를 정하지 못하거나 계획적으로 행동하지 못합니다. 
- 물건을 잃어버리거나 제자리에 놓지 않습니다.
 
④ 불안정함 혹은 끊임없는 활동
- 가만히 앉아 있는 것을 어려워합니다.
- 자극적이고 흥분되는 일을 추구합니다.
- 동시에 여러 가지 일을 합니다.
- 쉽게 지루해합니다.
 
⑤ 충동성
- 다른 사람의 대화에 자주 끼어듭니다.
- 자제를 잘 못합니다.
- 무례하거나 부적절한 생각을 그대로 내뱉습니다.
- 결과를 고려하지 않고 돌발적으로 행동합니다.
- 중독의 위험이 있습니다.
 
⑥ 감정 조절의 어려움
- 자존감과 성취감이 낮습니다.
- 비판에 대해 과민 반응하며 쉽게 좌절합니다.
- 감정 기복이 심하고 조급합니다.
- 예민하고 폭발적으로 화를 냅니다. 
진단
ADHD는 문진과 증세를 종합하여 진단합니다. 최근 스탠퍼드대에서 기능자기공명영상(fMRI)을 통해 ADHD를 진단하는 방법을 개발했다고 발표했습니다. 하지만 아직 검사실에서 정확하게 ADHD 환자 여부를 감별하는 진단법은 없습니다. 
 
성인 ADHD를 진단할 때 쉽게 사용할 수 있는 '성인 ADHD 자가보고척도(ASRS)'가 있습니다. 선별 질문에서 검게 칠한 부분에 체크한 문항이 4개 이상이면 추가적인 검사가 필요합니다. 
치료
ADHD 치료 방법은 약물 치료와 인지행동 치료로 구분됩니다. 성인에게는 사회적 관계 기술을 개선하기 위한 인지행동 치료를 많이 사용합니다. 성인 ADHD 환자는 스케줄러를 통해 체계적으로 일정을 관리하여 업무 효율과 집중력을 높일 수 있도록 도와줍니다. 감정 조절 훈련을 시행하여 화가 나고 감정 조절이 되지 않을 때 자신의 감정을 말로 표현하고 분노를 조절하는 방법을 익히도록 합니다. 
 
약물 치료에는 중추신경자극제인 메칠페니데이트(methylphenidate) 계통의 약물을 복용해 주의집중력을 호전시키는 방법이 있습니다. 다른 치료제에는 아토목세틴, 클로니딘, 부프로피온 등의 약제가 있습니다. 최근에는 뇌의 니코틴 수용체와 관련된 약을 개발하려는 연구가 활발하게 이루어지고 있습니다.
 
가족이나 동료들이 환자의 ADHD 증상이 개선될 수 있도록 이해하고 배려하여, 환자의 행동 교정을 유도하는 것이 바람직합니다. 
경과
ADHD가 치료되지 않고 지속되면 많은 문제가 발생합니다. 성인은 학교와 부모님의 통제를 벗어나 활동 반경이 넓어지며, 그에 맞추어 더 다양한 책임과 능력이 요구됩니다. ADHD 환아가 성인이 되면 사회에서는 아무도 그를 도와주거나 그의 행동에 책임져주지 않습니다. 
 
성인 ADHD 환자는 앞서 언급한 증상으로 인해 학업, 직장, 가정 등 일상생활 전반에서 기능 저하를 겪습니다. WHO는 전 세계적으로 발생하는 무단결근이나 업무 효율의 저하의 중요한 원인 중 하나로 ADHD를 꼽았습니다. 
 
ADHD 환자가 제대로 치료받지 않고 성인이 되면, 알코올 중독, 니코틴 중독, 게임 및 인터넷 중독, 핸드폰 중독에 쉽게 빠진다는 연구도 있습니다. 
주의사항
증세가 가벼운 환자는 증세를 스스로 조절합니다. 한꺼번에 여러 가지 일에 집중하는 '병행업무수행(multitasking)'은 현대 사회에서 재능으로 간주됩니다. 이 때문에 성인 ADHD 환자 중에는 동시에 여러 일에 집중하면서 성취해내는 사람도 있습니다. 하지만 많은 환자에게 '자기 조절'은 짐이 될 수 있습니다. 긴 회의를 참고 견디며 남이 말하는 것을 방해하지 않을 수는 있지만, 자기 조절에 너무 많은 에너지를 사용하느라 회의 내용이나 다른 사람이 말한 내용에 대해 신경을 쓰지 못하는 경우가 있습니다.
 
성인 ADHD의 증세는 매우 다양한 형태로 나타나므로, 신중하게 진단 및 치료해야 합니다. '혹시 성인 ADHD가 아닐까?'라는 의문이 든다면, 인터넷, 책 등을 통해 혼자서 판단하지 말고 전문가와 상담하여 정확하게 진단, 치료받아야 합니다. 
 
성인 ADHD 환자에게 유용한 습관
① 메모가 가능한 노트나 수첩, 스마트폰을 항시 휴대합니다.
② 주변에 휴지통과 정리함을 여러 개 배치합니다.
③ 열쇠, 전화기, 지갑 등의 물건을 담는 보관함을 항상 같은 위치에 두고 사용합니다.
④ 반복되는 실수를 파악하고 동일한 실수를 하지 않도록 주의합니다.
⑤ 주무를 수 있는 물건을 주머니에 항상 소지하면서 불안과 분노가 생길 때마다 이를 사용하여 감정을 조절합니다.  
"""
"""
textrank = TextRank(text)

sentence_num = int(0.1 * len(textrank.tokenized_sentence))

if sentence_num < 1:
    sentence_num = 1
textrank.analyze(number=int(sentence_num))
print("<요약>")
num=1
for n in range(int(sentence_num)):
    print("%d. "%num, textrank.summary[n])
    num+=1

keywords = textrank.keywords
numbers = textrank.numbers
ans_candidate = keywords
filtered_sentences = textrank.filtered_sentence
summary = textrank.summary

print("\n<중요단어들>")
print(keywords)
qg = question_generation(summary, ans_candidate)
print("\n<질문 생성>")
num=1
for i,j in qg.items():
    print("%d. "%num,i)
    print("답: ",j)
    num+=1
"""

Todo = Namespace('Todo')
@Todo.route('')
class TodoPost(Resource):
    def post(self):
        global textrank
        txt = request.json.get('data')
        textrank=TextRank(txt)
        sentence_num = int(0.1 * len(textrank.tokenized_sentence))
        if sentence_num < 1:
            sentence_num = 1
        textrank.analyze(number=int(sentence_num))
        num=1
        totalsum="<SUMMARY>\n"#api용
        for n in range(int(sentence_num)):
            print("%d. "%num, textrank.summary[n])
            totalsum+=str(num)+". "+textrank.summary[n]+"\n"
            num+=1
        keywords = textrank.keywords
        numbers = textrank.numbers
        ans_candidate = keywords
        filtered_sentences = textrank.filtered_sentence
        summary = textrank.summary
        totalkey="<KEYWORDS>\n"#api용
        for words in keywords:
            totalkey+="["+words+"]\n"
    
        qg = question_generation(summary, ans_candidate)
        totalq="<BLANK QUIZ>\n"#api용
        qnum=1
        for i,q in qg.items():
            totalq+=str(qnum)+"번째 질문: "+str(i)+"\n"+"답: "+str(q)+"\n"
            qnum+=1
        newtext=totalsum+totalkey+totalq
        return {
            #'data':newtext,
            'summary':summary,
            'keywords':keywords,
            'blank quiz':qg
        }


# In[ ]:




