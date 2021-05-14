import os
import re
import pandas as pd
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from bs4 import BeautifulSoup
# nltk.download('wordnet')
# nltk.download('stopwords')

dir = os.path.dirname(os.path.realpath(__file__))
train = pd.read_csv(dir+'./labeledTrainData.tsv', header=0, sep="\t", quoting=3)
test = pd.read_csv(dir+'./testData.tsv', header=0, sep="\t", quoting=3)

# print(train.info())
# print(test.info())
# print(train['sentiment'].value_counts())



'''
# 데이터 정제 Data Cleaning and Text Preprocessing

# pip3 install beautifulsoup4\
# pip install html5lib
# from bs4 import BeautifulSoup
example = BeautifulSoup(train['review'][0],'html5lib')
# print(train['review'][0][:700])
# print(example.get_text())       # <br \>과 같은 html 태그들이 사라짐

# import re
letters_only = re.sub('[^a-zA-Z]',' ',example.get_text())   # 특수문자를 빈칸으로 대체
# print(letters_only)

lower_case = letters_only.lower()
words = lower_case.split()
# print(words)    # 소문자로 바꾼 후, 단어단위로 토큰화





# 불용어 제거 (Stopword Removal)
# import nltk
# from nltk.corpus import stopwords
# nltk.download('stopwords')
# print(len(words)) # 437개 단어
words = [w for w in words if not w in stopwords.words('english')]
# print(len(words)) # 불용어 제거 후 219개 단어
# print(words)




# 어간 추출 - 어형이 변형된 단어로부터 접사 등을 제거하고 그 단어의 어간을 분리해내는 것
# from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer('english')
words = [stemmer.stem(w) for w in words]
# print(words)



# from nltk.stem import WordNetLemmatizer
# nltk.download('wordnet')
wordnet_lem = WordNetLemmatizer()
words = [wordnet_lem.lemmatize(w) for w in words]
# print(words[:10])
'''

def review_to_words(raw_review):
    review_text = BeautifulSoup(raw_review, "html.parser").get_text()
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    words = letters_only.lower().split()
    stops = set(stopwords.words("english"))
    meaningful_words = [w for w in words if not w in stops]
    stemmer = SnowballStemmer('english')
    stemming_words = [stemmer.stem(w) for w in meaningful_words]
    return(" ".join(stemming_words))

# 0. def로 함수 선언 
# 1. HTML 제거
# 2. 영문자가 아닌 문자는 공백으로 변환
# 3. 소문자로 전체 변환
# 4. 파이썬에서는 리스트보다 세트로 찾는게 훨씬 빠르다. stopwods를 세트로 변환
# 5. Stopwords 불용어 제거
# 6. Stemming으로 어간추출
# 7. 공백으로 구분된 문자열로 결합하여 결과 반환
# test :
# clean_review = review_to_words(train["review"][0])
# print(clean_review)



from multiprocessing import Pool  ## multiprocessing을 사용하면 복잡하고 오래걸리는 작업을 별도의 프로세스를 생성 후 # 병렬처리해서 보다 빠른 응답처리 속도를 기대할 수 있는 장점이 있다. # 출처: https://gist.github.com/yong27/7869662

def _apply_df(args):
    df, func, kwargs = args
    return df.apply(func, **kwargs)

def apply_by_multiprocessing(df, func, **kwargs):
    # 키워드 항목 중 workers 파라미터를 꺼냄
    workers = kwargs.pop("workers")
    # 위에서 가져온 workers 수로 프로세스 풀을 정의
    if __name__ == '__main__':    
        pool = Pool(processes = workers)
    # 실행할 함수와 데이터프레임을 워커의 수 만큼 나눠서 작업
        result = pool.map(_apply_df, [(d, func, kwargs)
                                 for d in np.array_split(df, workers)])
        pool.close()
    #작업 결과를 합쳐서 반환
        return pd.concat(list(result))

clean_train_reviews = apply_by_multiprocessing(train["review"],review_to_words, workers = 4)
# clean_test_reviews = apply_by_multiprocessing(test["review"],review_to_words, workers = 4)
# print(clean_train_reviews)
# print(clean_test_reviews)





# 워드 클라우드
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

def displayWordCloud(data = None, backgroundcolor = "black", width = 800, height = 600):
    wordcloud = WordCloud(stopwords = STOPWORDS,                         background_color = backgroundcolor,                         width = width, height = height).generate(data)
    plt.figure(figsize = (15, 10))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()
displayWordCloud(clean_train_reviews)
