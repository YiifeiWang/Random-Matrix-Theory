import numpy as np
import numpy.linalg as la

def GOE(N):
	H1 = np.triu(np.random.normal(scale=1,size=(N,N)))
	H = H1+H1.T
	H[range(N),range(N)] /=np.sqrt(2)
	return H

def GUE(N):
	H0 = np.random.normal(scale=np.sqrt(0.5),size=(N,N))
	H1 = np.triu(H0,k=1)
	H2 = np.tril(H0,k=-1)
	H3 = np.diag(np.diag(H0))*np.sqrt(2)
	H = H1+H1.T+np.complex(0,1)*H2+np.complex(0,-1)*H2.T+H3;
	return H

def Wigner_Real(N):
	H1 = np.triu(np.random.normal(scale=1,size=(N,N)))
	H = H1+H1.T
	H[range(N),range(N)] /=2
	return H

def Wigner_Complex(N,alpha):
	H0 = np.random.normal(scale=np.sqrt(1),size=(N,N))
	H1 = np.triu(H0,k=1)
	H2 = np.tril(H0,k=-1)
	H3 = np.diag(np.diag(H0))
	H = np.sqrt(alpha)*(H1+H1.T)+np.sqrt(1-alpha)*(np.complex(0,1)*H2+np.complex(0,-1)*H2.T)+H3;
	return H

def space(H):
	N = H.shape[0]
	eigs = np.real(la.eigvalsh(H))
	eigs.sort()
	space_H = eigs[1:]-eigs[:-1]
	return space_H

def mean_space(N, sample = 200, mtype = 'GOE'):
	m_space = np.zeros(N-1)
	for i in range(sample):
		if mtype == 'GOE':
			m_space += space(GOE(N))
		if mtype == 'GUE':
			m_space += space(GUE(N))
	m_space/=sample
	return m_space

def multi_space(N, sample = 200, mtype = 'GOE'):
	m_space = np.zeros([sample,N-1])
	for i in range(sample):
		if mtype == 'GOE':
			m_space[i] = space(GOE(N))
		if mtype == 'GUE':
			m_space[i] = space(GUE(N))
	return m_space