from gensim.corpora import WikiCorpus
from pyhanlp import *
import codecs
import re
import pickle
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

"""
note:
    reference: https://blog.csdn.net/kaoyala/article/details/79090156
    
1、下载wiki中文语料: http://download.wikipedia.com/zhwiki/latest/zhwiki-latest-pages-articles.xml.bz2
2、data_process() 读取中文wiki语料库，并解析提取xml中的内容
3、安装opencc用于中文的简繁替换
    建议安装：exe的版本，链接：https://bintray.com/package/files/byvoid/opencc/OpenCC
    下载版本：opencc-1.0.1-win64.7z， 解压
4、步骤2得到zhwiki-articles.txt，将其拷贝到opencc解压的文件夹
5、cmd进入opencc解压目录，run: opencc -i zhwiki-articles.txt -o zhwiki-articles-jian.txt -c t2s.json
   此文件夹下出现zhwiki-articles-jian.txt的简体版wiki语料
6、分词、去停用词
    分词：HanLp segment
    stopwords:  https://pan.baidu.com/s/1i6fY9Lj 密码: 4s5c
7、训练词向量
"""


class wiki_word2vector(object):
    """
    origin_path: wiki corpus download from web
    txt_path: the txt data though function data_process() [extract txt content from xml file]
    stopwords_path: stop words path
    model_path: the path to save trained word2vector model
    jan_txt_path: the path of txt data processed with opencc
    segment_txt_path: the path of txt data processed segment and took out stopwords
    stopwords: boolean type, whether take out stopwords or not
    nr_path: the path to save people name
    ns_path: the path to save place name
    """
    def __init__(self,
                 origin_path="zhwiki-latest-pages-articles.xml.bz2",
                 txt_path="zhwiki-articles.txt",
                 stopwords_path="stopword.txt",
                 model_path=os.path.join("./model", "wiki_word2vector.model"),
                 jan_txt_path="zhwiki-articles-jian.txt",
                 segment_txt_path="zhwiki-articles-segment.txt",
                 stopwords=True,
                 nr_path="nr_set.pkl",
                 ns_path="ns_set.pkl"
                 ):

        self.origin_path = origin_path
        self.txt_path = txt_path
        self.stopwords_path = stopwords_path
        self.jan_txt_path = jan_txt_path
        self.segment_txt_path = segment_txt_path
        self.stopwords = stopwords
        self.nr_path = nr_path
        self.ns_path = ns_path
        self.model_path = model_path

    def data_process(self):
        """
        extract txt content from xml file
        """

        space = " "
        i = 0
        output = open(self.txt_path, 'w', encoding='utf-8')
        wiki = WikiCorpus(self.origin_path, lemmatize=False, dictionary={})
        for text in wiki.get_texts():
            output.write(space.join(text) + "\n")
            i = i + 1
            if i % 10000 == 0:
                print('Saved ' + str(i) + ' articles')
        output.close()
        print('Finished Saved ' + str(i) + ' articles')

    def create_stop_list(self):
        """
        load stopwords
        """
        print('load stopwords...')
        stoplist = [line.strip() for line in codecs.open(self.stopwords_path, 'r', encoding='utf-8').readlines()]
        stopwords = {}.fromkeys(stoplist)
        return stopwords

    def is_Alpha(self, word):
        """
        filter english
        """
        try:
            return word.encode('ascii').isalpha()
        except UnicodeEncodeError:
            return False

    def segment(self):
        """
        segment for corpus and take out stopwords
        """
        stopwords = self.create_stop_list()
        NLPTokenizer = JClass("com.hankcs.hanlp.tokenizer.NLPTokenizer")
        nr_set = set()  # create name set to save name
        ns_set = set()  # create place name set to save them
        with open(self.segment_txt_path, "w", encoding="utf-8") as out_f:
            with open(self.jan_txt_path, "r", encoding="utf-8") as f:
                counter = 0
                for line in f:
                    line = re.sub("\s", "", line)  # 去掉空格(也去掉了换行符)
                    for term in NLPTokenizer.segment(line):
                        # filter the english word, such as "hello","world" .etc, However it doesn't work for "heool222"
                        if self.is_Alpha(term.word):
                            continue
                        if self.stopwords and term.word in stopwords:
                            continue
                        if len(str(term.word)) > 5:
                            continue
                        if str(term.nature) == "nr":
                            nr_set.add(term.word)
                            out_f.write(term.word)
                            out_f.write("  ")
                        elif str(term.nature) == "ns":
                            ns_set.add(term.word)
                            out_f.write(term.word)
                            out_f.write("  ")
                        else:
                            out_f.write(term.word)
                            out_f.write("  ")

                    out_f.write("\n")
                    counter += 1
                    if counter % 10000 == 0:
                        print("Have processed 【{}】 sentences!".format(counter))

        with open(self.nr_path, 'wb') as f:
            pickle.dump(nr_set, f)
        with open(self.ns_path, 'wb') as f:
            pickle.dump(ns_set, f)
        print("segment complete.")

    def train_word2vector(self):
        print("Begin train word_vector from {}".format(self.segment_txt_path))
        model = Word2Vec(LineSentence(self.segment_txt_path),
                         size=256,
                         window=5,
                         min_count=1,
                         workers=10,
                         iter=10)
        model.save(self.model_path)
        print("the word2vector model has complete, and save the model to {}".format(self.model_path))


if __name__ == "__main__":
    wiki = wiki_word2vector()
    # step 2
    # wiki.data_process()
    # step 6
    # wiki.segment()
    # step 7
    wiki.train_word2vector()
