import numpy as np
import matplotlib.pyplot as plt

eps = 1e-16

def eulerd(p1, p2):
	return np.sqrt(np.sum((p1-p2) ** 2, 1))

#K-Means
def init(data, K):
	nsample = data.shape[0]
	D = data.shape[1]
	uidx = np.random.permutation(nsample)
	u = data[uidx[0:K], :]

	ploss = -100
	distanceMatrix = np.zeros((nsample, 4))
	for i in range(0, K):
		distanceMatrix[:, i] = eulerd(data, np.tile(u[i,:], (nsample, 1)))

	classid = np.argmin(distanceMatrix, 1)
	loss = np.sum(distanceMatrix[classid])
	while abs(loss - ploss) > eps:
		for i in range(0, K):
			maski = np.where(classid == i)
			maski = maski[0]
			ui = np.sum(data[maski, :], 0) / maski.shape[0]
			distanceMatrix[:, i] = eulerd(data, np.tile(ui, (nsample, 1)))
		classid = np.argmin(distanceMatrix, 1)
		ploss = loss
		loss = np.sum(distanceMatrix[classid])

	plt.scatter(data[:, 0], data[:, 1], c=classid-1, marker='s')
	plt.show()
	GMMpara = []
	for i in range(0, K):
		idx = np.where(classid == i)
		idx = idx[0]
		ui = np.sum(data[idx, :], 0) / idx.shape[0]
		Zi = np.dot(np.transpose(data-ui), data-ui) / nsample
		GMMpara.append((1./K, ui, Zi))
	return GMMpara

def calGausian(para, x, D):
	return para[0] * ((2*np.pi) ** (-D/2)) * (np.linalg.det(para[2]) ** -0.5) * np.exp(-0.5 * np.dot(np.dot(x - para[1], np.linalg.inv(para[2])), np.transpose(x - para[1])))

def loglikelihood(data, GMMpara, K):
	K = len(GMMpara)
	nsample = data.shape[0]
	D = float(data.shape[1])
	sumll = 0.0
	for i in range(0, nsample):
		sumlli = 0.0
		for j in range(0, K):
			sumlli += calGausian(GMMpara[j], data[i, :], D)
		sumll += np.log(sumlli)
	return sumll

'''
@input:  initGMMpara = [(nk, uk, Zk)]

@return: GMMpara = [(nk, uk, Zk)]
'''
def train(initGMMpara, data, K):
	D = data.shape[1]
	nsample = data.shape[0]
	pllh = -100000.0
	llh = loglikelihood(data, initGMMpara, K)
	GMMpara = initGMMpara
	while abs(llh - pllh) > eps:
		gamma = np.zeros((nsample, K))

		for i in range(0, nsample):
			sumll = 0.0
			for j in range(0, K):
				gamma[i, j] = calGausian(GMMpara[j], data[i, :], D)
				sumll += gamma[i, j]
			gamma[i, :] /= sumll
		for i in range(0, K):
			gi = gamma[:, i].reshape(nsample, 1)
			NK = np.sum(gi)
			nk = NK / np.sum(np.sum(gamma))
			uk = np.sum(gi* data, 0) / NK
			Zk = np.dot(np.transpose(gi * (data - uk)), data - uk) / NK
			GMMpara[i] = (nk, uk, Zk)
		pllh = llh
		llh = loglikelihood(data, GMMpara, K)
		print llh, pllh
	return GMMpara




if __name__ == '__main__':
	data = np.loadtxt('data.txt', delimiter=' ')
	K = 4
	initGMMpara = init(data, K)
	GMMpara = train(initGMMpara, data, K)
	print GMMpara