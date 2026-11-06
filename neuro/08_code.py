import math



def neuron(inputs, weights, bias) -> int:
	'''один нейрон: взвешенная сумма входов'''
	total = bias
	for x, w in zip(inputs, weights):
		total += x * w
	return total

def relu(x) -> float | int:
	'''Обнуление отрицательного'''
	return x if x > 0 else 0

def matvec(matrix, vector) -> list:
	'''умножение матрицы на вектор по правилу строка на вектор'''
	out = []
	for row in matrix:
		sum_ = 0
		for a, b in zip(row, vector):
			sum_ += a * b
		out.append(sum_)
	return out

def softmax(logits) -> softmax:
	'''превращает сырые оценки в вероятности (сумма = 1)'''
	m = max(logits)
	exps = [math.exp(x - m) for x in logits]
	total = sum(exps)
	return [e / total for e in exps]

def tokenize(text, vocab) -> list:
	'''простой токенизатор'''
	tokens = []
	i = 0
	while i < len(text):
		for j in range(len(text), i, -1):
			piece = text[i:j]
			if piece in vocab:
				tokens.append(vocab[piece])
				i = j
				break
		else:
			i += 1
	return tokens


def attention(query, keys, values) -> tuple:
	'''оценка скалярное произведение запроса на каждый ключ'''
	scores = []
	for k in keys:
		dot = 0
		for a, b in zip(query, k):
			dot += a * b
		scores.append(dot)
	# оценки в вероятности
	weights = softmax(scores)
	# взвешенная сумма значений
	out = [0.0 for _ in values[0]]
	for w, v in zip(weights, values):
		for i in range(len(out)):
			out[i] += w * v[i]
	return weights, out


# ввод нейронов
print(neuron([25, 90], [0.1, 0.8], -50))
print(relu(neuron([25, 90], [0.1, 0.8], -50)))

# матрица на вектор - усиливает второй вход, первый не трогает
print(matvec([[1, 0], [0, 2]], [2, 3]))

# логиты в вероятности
print(softmax([2, 1, 0.1]))

# токенизация по словарю
vocab = {"привет": 1, "мир": 2, " ": 3}
print(tokenize("привет мир", vocab))

query = [1.0, 0.0]
keys = [[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]]
values = [[10.0, 0.0], [0.0, 20.0], [5.0, 5.0]]

weights, out = attention(query, keys, values)
print(weights)
print(out)
