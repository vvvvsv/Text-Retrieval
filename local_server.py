from threading import Thread
import socket
import sys
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler

class LocalServer(object):
    """
    初始化，读取去重后的文本
    读取词典
    读取TFIDF矩阵并降维得到文章向量
    读取相似词和相似度
    """
    def __init__(self, host, port, alpha=0.5):
        self.alpha = alpha
        self.address = (host, port)
        self.news = pd.read_csv('./data/processed_news.csv', index_col=0)
        with open('./data/vocab.txt', 'r') as f:
            self.vocab = [s.strip() for s in f.readlines()]
        self.tfidf = np.load('./data/tfidf.npy')
        tmp = normalize(self.tfidf, norm='l2')
        tmp = PCA(n_components=1000).fit_transform(tmp)
        tmp = normalize(tmp, norm='l2')
        self.pca_tfidf = tmp
        self.syn_words = np.load('./data/synonym_words.npy')
        self.syn_score = np.load('./data/synonym_score.npy')

    """
    在服务器端实现合理的并发处理方案，使得服务器端能够处理多个客户端发来的请求
    """
    def run(self):
        try:
            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind(self.address)
            server.listen(5)
        except socket.error as msg:
            print(msg)
            sys.exit(1)
        print('Server initialization complete.')
        print('Waiting for connection...')

        while 1:
            conn, addr = server.accept()
            print(conn, 'connected!')
            t = Thread(target = self.search, args=(conn, addr))
            t.start()
        return

    """
    实现文本检索，以及服务器端与客户端之间的通信

    1. 接受客户端传递的数据， 例如检索词
    2. 调用检索函数，根据检索词完成检索
    3. 将检索结果发送给客户端，具体的数据格式可以自己定义
    """
    def search(self, conn, addr):
        conn.send(("Connected!").encode())
        raw_words, take_intersection, fuzzy_search = eval(conn.recv(1024).decode())

        # words = {word:score, word:score}
        # score 为每个单词的评分，代表其重要程度
        # 若为初始的检索词，则 score=1
        # 若为检索词的相似词，则 score为该词与各检索词相似度的最大值
        words = dict()
        for word in raw_words:
            try:
                words[self.vocab.index(word)] = 1.0
            except ValueError:
                continue

        if len(words)==0:
            conn.send('[]'.encode())
            print(conn, 'done.')
            sys.stdout.flush()
            conn.close()
            return


        # 如果设置了模糊匹配，则在 words 中加入相似词
        if fuzzy_search==1:
            syn = []
            for word_id, _ in words.items():
                syn.extend(zip(self.syn_words[word_id], self.syn_score[word_id]))
            for word_id, score in syn:
                if word_id in words:
                    words[word_id] = max(words[word_id], score)
                else:
                    words[word_id] = score

        # 统计包含 words 中单词至少一次的文章，保存在 documents 中，并计算 score1 评分
        # 文章的 score1 评分为它包含的单词的评分*单词在该文章中的TFIDF值的和
        sum = np.zeros_like(self.tfidf[:, 0])
        for word_id, score in words.items():
            sum += self.tfidf[:, word_id] * score

        documents = []
        scores1 = []
        scores2 = []
        for id, score in enumerate(sum):
            if score > 1e-6:
                documents.append(id)
                scores1.append(score)

        if len(documents)==0:
            conn.send('[]'.encode())
            print(conn, 'done.')
            sys.stdout.flush()
            conn.close()
            return


        # 使用 HITS 算法为各文章评分，并记录在 scores2 中
        # 加入每篇文章的相似文章
        sim_documents = []
        for id in documents:
            sim_documents.extend(self.calc_sim_news(id))

        sim_documents.extend(documents)
        sim_documents = sorted(list(set(sim_documents)))

        pos_dict = dict()
        for i,id in enumerate(sim_documents):
            pos_dict[id]=i

        # 计算集合中文章两两之间相似度，得到一对称矩阵 sim_mat
        sim_mat = self.calc_sim_mat(sim_documents)
        # 迭代求该矩阵的主特征向量
        prin_eigv = self.calc_prin_eigv(sim_mat, 1e-6)
        # score2 得分为 sim_mat 中的该行与主特征向量的内积
        for id in documents:
            score = np.dot(prin_eigv, sim_mat[pos_dict[id]])
            scores2.append(score)

        # 将 scores1 和 scores2 分别标准化并加权求和，scores1 的权值为 alpha，scores2 的权值为 1-alpha
        scores = self.alpha * StandardScaler().fit_transform(np.array(scores1).reshape(-1, 1)) + \
                    (1 - self.alpha) * StandardScaler().fit_transform(np.array(scores2).reshape(-1, 1))
        scores = scores.reshape(-1)

        # 按最终得分排序
        documents = np.array(documents)
        documents = documents[np.argsort(-scores)]

        # 取出标题和文本并返回
        ret = []
        if take_intersection == 0:
            for id in documents:
                ret.append((self.news.loc[id, 'title'], self.news.loc[id, 'body']))
        else:
            # 若用户要求必须包含全部检索词，则只返回包含所有检索词的文本
            for id in self.intersection_documents(raw_words, documents):
                ret.append((self.news.loc[id, 'title'], self.news.loc[id, 'body']))

        conn.send(repr(ret).encode())
        print(conn, 'done.')
        sys.stdout.flush()
        conn.close()

        return

    """
    实现文本检索，以及服务器端与客户端之间的通信

    1. 接受客户端传递的数据， 例如检索词
    2. 调用检索函数，根据检索词完成检索
    3. 将检索结果发送给客户端，具体的数据格式可以自己定义
    """
    def intersection_documents(self, raw_words, documents):
        ret = []
        words = []
        for word in raw_words:
            try:
                words.append(self.vocab.index(word))
            except ValueError:
                return []

        for i in documents:
            if np.where(self.tfidf[i, words] > 1e-6, 1, 0).sum() == len(words):
                ret.append(i)
        return ret

    """
    计算与编号为id的文章相似的文章，并以列表形式返回
    """
    def calc_sim_news(self, id):
        cos_sim = np.dot(self.pca_tfidf, self.pca_tfidf[id])
        return list(np.argwhere(cos_sim > 0.2).reshape((-1)))

    """
    计算sim_documents中的文章的相似度矩阵sim_mat并返回
    其中sim_mat[i,j]=k 表示sim_documents[i]与sim_documents[j]的相似度为k
    """
    def calc_sim_mat(self, sim_documents):
        fea_mat = self.pca_tfidf[sim_documents]
        sim_mat = np.matmul(fea_mat, fea_mat.T)
        return sim_mat

    """
    计算矩阵mat的主特征向量并返回
    """
    def calc_prin_eigv(self, mat, eps):
        x = np.random.rand(mat.shape[1])
        while True:
            x1 = np.dot(mat, x)
            x1 = x1 / np.linalg.norm(x1)
            if np.abs(x1-x).max() < eps:
                break
            x = x1
        return x

if __name__ == '__main__':
    server = LocalServer("0.0.0.0", 1234, 0.5)
    server.run()