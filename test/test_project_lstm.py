import sys

sys.path.append("C:/Users/z/Desktop/erud")

import erud
from erud.upf.adam import adam
from erud.upf.norm import norm
from erud._utils import useGPU
if useGPU :
    import cupy as cp
import numpy as np

import emoji
import csv


def read_glove_vecs(glove_file):
    with open(glove_file, 'r', encoding='utf-8') as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
        
        i = 1
        words_to_index = {}
        index_to_words = {}
        for w in sorted(words):
            words_to_index[w] = i
            index_to_words[i] = w
            i = i + 1
    return words_to_index, index_to_words, word_to_vec_map


def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)]
    return Y


def read_csv(filename):
    phrase = []
    emoji = []

    with open (filename) as csvDataFile:
        csvReader = csv.reader(csvDataFile)

        for row in csvReader:
            phrase.append(row[0])
            emoji.append(row[1])

    X = np.asarray(phrase)
    Y = np.asarray(emoji, dtype=int)

    return X, Y

def sentenes_to_indices(X, word_to_index, max_len) :
     
    m = X.shape[0]  # 训练集数量
    # 使用0初始化X_indices
    X_indices = np.zeros((m, max_len))
    
    for i in range(m):
        # 将第i个居住转化为小写并按单词分开。
        sentences_words = X[i].lower().split()
        
        # 初始化j为0
        j = 0
        
        # 遍历这个单词列表
        for w in sentences_words:
            # 将X_indices的第(i, j)号元素为对应的单词索引
            X_indices[i, j] = word_to_index[w]
            
            j += 1
            
    return X_indices

def sentenes_to_vec(X, word_to_vec_map, max_len) :

    m = X.shape[0]  # 训练集数量
    # 使用0初始化X_indices
    X_indices = np.zeros((m, max_len, 50))
    
    for i in range(m):
        # 将第i个居住转化为小写并按单词分开。
        sentences_words = X[i].lower().split()
        
        # 初始化j为0
        j = 0
        
        # 遍历这个单词列表
        for w in sentences_words:
            # 将X_indices的第(i, j)号元素为对应的单词索引
            X_indices[i, j, :] = word_to_vec_map[w]
            
            j += 1
            
    return X_indices

def getXY():
	path = __file__[:__file__.rfind('\\')]
	X_train, Y_train = read_csv(path + '/datasets/train_emoji.csv')
	X_test, Y_test = read_csv(path + '/datasets/test.csv')

	maxLen = len(max(X_train, key=len).split())
	Y_oh_train = convert_to_one_hot(Y_train, C=5)
	Y_oh_test = convert_to_one_hot(Y_test, C=5)

	word_to_index, index_to_word, word_to_vec_map = read_glove_vecs(__file__[:__file__.rfind('\\')] + '/datasets/glove.6B.50d.txt')

	# X_train_indices = sentenes_to_indices(X_train, word_to_index, maxLen)
	# X_test_indices = sentenes_to_indices(X_test, word_to_index, maxLen)
	X_train_indices = sentenes_to_vec(X_train, word_to_vec_map, maxLen)
	X_test_indices = sentenes_to_vec(X_test, word_to_vec_map, maxLen)

	return X_train, Y_train, X_test, Y_test, X_train_indices, Y_oh_train, X_test_indices, Y_oh_test
	
def no_test_lstm () :
	X_train, Y_train, X_test, Y_test, X_train_indices, Y_oh_train, X_test_indices, Y_oh_test = getXY()

	print(X_train_indices.shape)
	# print(X_train_indices[5, 2])
	print(Y_oh_train.shape)

	num_iterations = 70
	batches = int(X_train_indices.shape[0] / 32)
	rate = 0.003

	g = erud.nous(
          '''
	X:(32, 10, 50) scatter(1) -> loop t = 1 to 10 (
		x<t> matmul Wfx1 add (a<t - 1> matmul Wfa1) add bf1 -> sigmoid as Gf<t>
		x<t> matmul Wux1 add (a<t - 1> matmul Wua1) add bu1 -> sigmoid as Gu<t>
		x<t> matmul Wcx1 add (a<t - 1> matmul Wca1) add bc1 -> tanh => ct<t>
		Gf<t> mul c<t - 1> add (Gu<t> mul ct<t>) => c<t>
		x<t> matmul Wox1 add (a<t - 1> matmul Woa1) add bo1 -> sigmoid as Go<t>
		c<t> -> tanh -> mul Go<t> => a<t> -> matmul Way1 add by1 -> softmax(1)
	) -> 
    gather(1) ->
	dropout(0.8) -> 
    scatter(1) -> loop t = 21 to 30 (
		x<t> matmul Wfx2 add (a<t - 1> matmul Wfa2) add bf2 -> sigmoid as Gf<t>
		x<t> matmul Wux2 add (a<t - 1> matmul Wua2) add bu2 -> sigmoid as Gu<t>
		x<t> matmul Wcx2 add (a<t - 1> matmul Wca2) add bc2 -> tanh => ct<t>
		Gf<t> mul c<t - 1> add (Gu<t> mul ct<t>) => c<t>
		x<t> matmul Wox2 add (a<t - 1> matmul Woa2) add bo2 -> sigmoid as Go<t>
		c<t> -> tanh -> mul Go<t> => a<t>
	)

	a30 -> matmul Way2 add by2 -> softmax(1) ->

    cross_entropy Y:(32, 5) -> cost -> J:$$
'''
	).parse()
	'''
		X:(32, 10, 50) scatter(1) -> loop t = 1 to 10 (
		x<t> matmul Wfx1 add (a<t - 1> matmul Wfa1) add bf1 -> sigmoid as Gf<t>
		x<t> matmul Wux1 add (a<t - 1> matmul Wua1) add bu1 -> sigmoid as Gu<t>
		x<t> matmul Wcx1 add (a<t - 1> matmul Wca1) add bc1 -> tanh => ct<t>
		Gf<t> mul c<t - 1> add (Gu<t> mul ct<t>) => c<t>
		x<t> matmul Wox1 add (a<t - 1> matmul Woa1) add bo1 -> sigmoid as Go<t>
		c<t> -> tanh -> mul Go<t> => a<t> -> matmul Way1 add by1 -> softmax(1)
	) -> 
    gather(1) ->
	dropout(0.5) -> 
    scatter(1) -> loop t = 21 to 30 (
		x<t> matmul Wfx2 add (a<t - 1> matmul Wfa2) add bf2 -> sigmoid as Gf<t>
		x<t> matmul Wux2 add (a<t - 1> matmul Wua2) add bu2 -> sigmoid as Gu<t>
		x<t> matmul Wcx2 add (a<t - 1> matmul Wca2) add bc2 -> tanh => ct<t>
		Gf<t> mul c<t - 1> add (Gu<t> mul ct<t>) => c<t>
		x<t> matmul Wox2 add (a<t - 1> matmul Woa2) add bo2 -> sigmoid as Go<t>
		c<t> -> tanh -> mul Go<t> => a<t> -> matmul Way2 add by2 -> softmax(1) => yhat<t>
	)

	yhat30 -> dropout(0.5) -> softmax(1) ->

    cross_entropy Y:(32, 5) -> cost -> J:$$
    '''
	# print(g)

	g.setData('Wfx1', np.random.randn(50, 128) * 0.01)
	g.setData('Wfa1', np.random.randn(128, 128) * 0.01)
	g.setData('bf1', np.zeros((128)))
	g.setData('Wux1', np.random.randn(50, 128) * 0.01)
	g.setData('Wua1', np.random.randn(128, 128) * 0.01)
	g.setData('bu1', np.zeros((128)))
	g.setData('Wcx1', np.random.randn(50, 128) * 0.01)
	g.setData('Wca1', np.random.randn(128, 128) * 0.01)
	g.setData('bc1', np.zeros((128)))
	g.setData('Wox1', np.random.randn(50, 128) * 0.01)
	g.setData('Woa1', np.random.randn(128, 128) * 0.01)
	g.setData('bo1', np.zeros((128)))
	g.setData('Way1', np.random.randn(128, 50) * 0.01)
	g.setData('by1', np.zeros((50)))

	g.setData('Wfx2', np.random.randn(50, 128) * 0.01)
	g.setData('Wfa2', np.random.randn(128, 128) * 0.01)
	g.setData('bf2', np.zeros((128)))
	g.setData('Wux2', np.random.randn(50, 128) * 0.01)
	g.setData('Wua2', np.random.randn(128, 128) * 0.01)
	g.setData('bu2', np.zeros((128)))
	g.setData('Wcx2', np.random.randn(50, 128) * 0.01)
	g.setData('Wca2', np.random.randn(128, 128) * 0.01)
	g.setData('bc2', np.zeros((128)))
	g.setData('Wox2', np.random.randn(50, 128) * 0.01)
	g.setData('Woa2', np.random.randn(128, 128) * 0.01)
	g.setData('bo2', np.zeros((128)))
	g.setData('Way2', np.random.randn(128, 5) * 0.01)
	g.setData('by2', np.zeros((5)))

	g.setUpdateFunc('Wfx1', adam(rate))
	g.setUpdateFunc('Wfa1', adam(rate))
	g.setUpdateFunc('bf1', adam(rate))
	g.setUpdateFunc('Wux1', adam(rate))
	g.setUpdateFunc('Wua1', adam(rate))
	g.setUpdateFunc('bu1', adam(rate))
	g.setUpdateFunc('Wcx1', adam(rate))
	g.setUpdateFunc('Wca1', adam(rate))
	g.setUpdateFunc('bc1', adam(rate))
	g.setUpdateFunc('Wox1', adam(rate))
	g.setUpdateFunc('Woa1', adam(rate))
	g.setUpdateFunc('bo1', adam(rate))
	g.setUpdateFunc('Way1', adam(rate))
	g.setUpdateFunc('by1', adam(rate))

	g.setUpdateFunc('Wfx2', adam(rate))
	g.setUpdateFunc('Wfa2', adam(rate))
	g.setUpdateFunc('bf2', adam(rate))
	g.setUpdateFunc('Wux2', adam(rate))
	g.setUpdateFunc('Wua2', adam(rate))
	g.setUpdateFunc('bu2', adam(rate))
	g.setUpdateFunc('Wcx2', adam(rate))
	g.setUpdateFunc('Wca2', adam(rate))
	g.setUpdateFunc('bc2', adam(rate))
	g.setUpdateFunc('Wox2', adam(rate))
	g.setUpdateFunc('Woa2', adam(rate))
	g.setUpdateFunc('bo2', adam(rate))
	g.setUpdateFunc('Way2', adam(rate))
	g.setUpdateFunc('by2', adam(rate))

	# g.setData('X', X_train_indices[0:32])
	# g.setData('Y', Y_oh_train[0:32])

	for epoch in range(num_iterations) :
		for batch in range(batches) :
			X = X_train_indices[batch * 32:batch * 32 + 32]
			Y = Y_oh_train[batch* 32:batch*32 + 32]

			g.setData('a0', np.zeros((32, 128)))
			g.setData('c0', np.zeros((32, 128)))
			g.setData('a20', np.zeros((32, 128)))
			g.setData('c20', np.zeros((32, 128)))

			g.setData('X', X)
			g.setData('Y', Y)

			g.fprop()
			g.bprop()
		
		if X_train_indices.shape[0] % 32 != 0 :
			u = X_train_indices.shape[0] % 32
			X = X_train_indices[-u:]
			Y = Y_oh_train[-u:]

			g.setData('a0', np.zeros((u, 128)))
			g.setData('c0', np.zeros((u, 128)))
			g.setData('a20', np.zeros((u, 128)))
			g.setData('c20', np.zeros((u, 128)))

			g.setData('X', X)
			g.setData('Y', Y)

			g.fprop()
			g.bprop()
		
		if epoch % 1 == 0 :
			print('Cost after {} iteration is {}'.format(epoch, g.getData('J')))
			# print(g.getData('Wya'))
	

	gtest = erud.nous(
          '''
	X:(132, 10, 50) scatter(1) -> loop t = 1 to 10 (
		x<t> matmul Wfx1 add (a<t - 1> matmul Wfa1) add bf1 -> sigmoid as Gf<t>
		x<t> matmul Wux1 add (a<t - 1> matmul Wua1) add bu1 -> sigmoid as Gu<t>
		x<t> matmul Wcx1 add (a<t - 1> matmul Wca1) add bc1 -> tanh => ct<t>
		Gf<t> mul c<t - 1> add (Gu<t> mul ct<t>) => c<t>
		x<t> matmul Wox1 add (a<t - 1> matmul Woa1) add bo1 -> sigmoid as Go<t>
		c<t> -> tanh -> mul Go<t> => a<t> -> matmul Way1 add by1 -> softmax(1)
	) -> 
    gather(1) ->
    scatter(1) -> loop t = 21 to 30 (
		x<t> matmul Wfx2 add (a<t - 1> matmul Wfa2) add bf2 -> sigmoid as Gf<t>
		x<t> matmul Wux2 add (a<t - 1> matmul Wua2) add bu2 -> sigmoid as Gu<t>
		x<t> matmul Wcx2 add (a<t - 1> matmul Wca2) add bc2 -> tanh => ct<t>
		Gf<t> mul c<t - 1> add (Gu<t> mul ct<t>) => c<t>
		x<t> matmul Wox2 add (a<t - 1> matmul Woa2) add bo2 -> sigmoid as Go<t>
		c<t> -> tanh -> mul Go<t> => a<t>
	)

	a30 -> matmul Way2 add by2 -> max_index(1) -> accuracy Y:(132, 1) -> J:$$
'''
	).parse()

	gtest.setData('Wfx1', g.getData('Wfx1'))
	gtest.setData('Wfa1', g.getData('Wfa1'))
	gtest.setData('bf1', g.getData('bf1'))
	gtest.setData('Wux1', g.getData('Wux1'))
	gtest.setData('Wua1', g.getData('Wua1'))
	gtest.setData('bu1', g.getData('bu1'))
	gtest.setData('Wcx1', g.getData('Wcx1'))
	gtest.setData('Wca1', g.getData('Wca1'))
	gtest.setData('bc1', g.getData('bc1'))
	gtest.setData('Wox1', g.getData('Wox1'))
	gtest.setData('Woa1', g.getData('Woa1'))
	gtest.setData('bo1', g.getData('bo1'))
	gtest.setData('Way1', g.getData('Way1'))
	gtest.setData('by1', g.getData('by1'))
	

	gtest.setData('Wfx2', g.getData('Wfx2'))
	gtest.setData('Wfa2', g.getData('Wfa2'))
	gtest.setData('bf2', g.getData('bf2'))
	gtest.setData('Wux2', g.getData('Wux2'))
	gtest.setData('Wua2', g.getData('Wua2'))
	gtest.setData('bu2', g.getData('bu2'))
	gtest.setData('Wcx2', g.getData('Wcx2'))
	gtest.setData('Wca2', g.getData('Wca2'))
	gtest.setData('bc2', g.getData('bc2'))
	gtest.setData('Wox2', g.getData('Wox2'))
	gtest.setData('Woa2', g.getData('Woa2'))
	gtest.setData('bo2', g.getData('bo2'))
	gtest.setData('Way2', g.getData('Way2'))
	gtest.setData('by2', g.getData('by2'))

	# 训练集样本数132
	gtest.setData('a0', np.zeros((132, 128)))
	gtest.setData('c0', np.zeros((132, 128)))
	gtest.setData('a20', np.zeros((132, 128)))
	gtest.setData('c20', np.zeros((132, 128)))
     
	gtest.setData('X', X_train_indices)
	gtest.setData('Y', Y_train)
	
	gtest.fprop()
	print('train accuracy: %s' %(gtest.getData('J')))


	# 测试集样本数56
	gtest.setData('a0', np.zeros((56, 128)))
	gtest.setData('c0', np.zeros((56, 128)))
	gtest.setData('a20', np.zeros((56, 128)))
	gtest.setData('c20', np.zeros((56, 128)))
     
	gtest.setData('X', X_test_indices)
	gtest.setData('Y', Y_test)

	gtest.fprop()
	print('test accuracy: %s' %(gtest.getData('J')))








if __name__ == '__main__' :
	test_lstm()