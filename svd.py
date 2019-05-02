import pandas as pd
import numpy as np
import random
import math

def train_validate_split(df):
    indices_of_rows = df.index.tolist()
    validate_size = int(len(indices_of_rows)*20/100)
    validate_indices = random.sample(population=indices_of_rows, k=validate_size)
    validate_df = df.loc[validate_indices]
    train_df = df.drop(validate_indices)
    return train_df,validate_df

# # Serial Dot Prosucts

def oneDProduct(A,B):
    if(len(A) != len(B)):
        print ("Vectors have unequal dimensions")
        return
    
    else:
        rslt = 0
        for i in range(0,len(A)):
            rslt += A[i] * B[i]

        return rslt

def initialiseResult(dim1,dim2):
    rslt = []
    for i in range(0,dim1):
        l = []
        for j in range(0,dim2):
            l.append(0)
        rslt.append(l)
    
    return rslt

def matMul(A,B):
    ## IF BOTH ARE 1D ARRAYS
    if( len(A[0]) == 1 and len(B[0]) == 1 ):
        rslt = oneDProduct(A,B)
        return rslt

    ## IF BOTH ARE 2D ARRAYS
    if( len(A[0]) != 1 and len(B[0]) != 1):

        if( len(A[0]) != len(B) ):
            print("Matrices must have compatible dimensions")
            return
        
        rslt = initialiseResult(len(A),len(B[0]))

        for i in range(0,len(A)):
            for j in range(0,len(B[0])):
                for k in range(0,len(A[0])):
                    rslt[i][j] += A[i][k] * B[k][j]

        return rslt

def dotProductScalar(A,B):
    if(not isinstance(B[0],list)):
        rslt = [0] * len(B)
        for i in range(0,len(B)):
            rslt[i] = A * B[i]

        return rslt

    rslt = initialiseResult(len(B),len(B[0]))
    for i in range(0,len(B)):
        for j in range(0,len(B[0])):
            rslt[i][j] = A * B[i][j]

    return rslt

def vectorMulti(A,B):
    if(len(A[0]) != len(B)):
        print ("Matrix and vector dimensions are not compatible")
        return

    rslt = [0] * len(A)
    for i in range(len(A)):
        for j in range(len(A[0])):
            rslt[i] += A[i][j] * B[j]

    return rslt

def vectorMulti2(A,B):
    if(len(A) != len(B)):
        print ("Vector and matrix dimensions not compatible")
        return

    rslt = [0] * len(B[0])
    for i in range(0,len(B[0])):
        for j in range(0,len(A)):
            rslt[i] += A[j] * B[j][i]
    
    return rslt

def dotProduct(A,B):
    Ashape = A.shape
    Bshape = B.shape

    if (len(Bshape) == 1 and len(Ashape) > 1):
        rslt = vectorMulti(A,B)
#         print (rslt)
        return rslt
    
    if (len(Ashape) == 1 and len(Bshape) > 1):
        rslt = vectorMulti2(A,B)
#         print (rslt)
        return rslt
    
    if (len(Ashape) > 1 and len(Bshape) > 1):
        rslt = matMul(A,B)
#         print (rslt)
        return rslt
    
    if (len(Ashape) == 1 and len(Bshape) == 1):
        rslt = oneDProduct(A,B)
#         print (rslt)
        return rslt


# # Serial EigenValue And Vector Calculation

# In[446]:


Eigen_values = []
eigenvectors_list = []


def checkDiagonal(arr):
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            if i == j:
                continue
            else:
                if abs(arr[i][j]) > 0.001:
                    return False
    return True

def qrFactorization(arr):
    temp = arr
    i = 0
    eigenvectors = np.identity(len(arr))
    while(True):
        Q,R = np.linalg.qr(temp)
        temp = np.dot(R, Q)
        eigenvectors = np.dot(eigenvectors, Q)
        if(checkDiagonal(temp)):
            print("Number of Factorizations: " + str(i+1))
            break
        else:
            i += 1
    for i in range(len(arr)):
        eigenvectors_list.append(eigenvectors[:, i].tolist())
    return temp

def printLambda(arr):
	count = 1
	for i in range(len(arr)):
		for j in range(len(arr[i])):
			if(i == j):
				temp = arr[i][j]
				if(abs(temp) < 0.000000000001):
						temp = 0
				Eigen_values.append(temp)
				print("Lamda"+str(count) +": " + str(temp))
				count += 1

def read():
    f = open('matrix.txt', 'r')
    temp = f.read().split('\n')
    arr = []
    for i in temp:
        if i == '':
            continue
        arr.append(i.split(" "))
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            arr[i][j] = (arr[i][j])
    return arr

# def eigen_vec(A,Eigen_values):
# 	Iden = np.identity(len(A))
# 	B = []
# 	for i in range(len(A)):
# 		B.append([])
# 		for j in range(len(A[0])):
# 			# print(type(A[i][j]))
# 			B[i].append(float(A[i][j]) - (Eigen_values[i] * Iden[i][j]))
	
# 	zeros = [0]*len(A)
# 	v = luSolve(B,zeros,len(A))
# 	print("vector",v)
# 	# return B

def main():
    arr = read()
    matrix = np.array(arr)
    print(matrix)
    printLambda(qrFactorization(arr))

    print(eigenvectors_list)
    # print(np.linalg.eig(matrix))
    # print(eigen_vec(matrix,Eigen_values))

if __name__ == '__main__':
    main()
    print(Eigen_values)


# # Rading MNIST Data Set

# In[416]:


df = pd.read_csv('train.csv',delimiter=',')
df.head()


# # Spliting Data into Train and Test

# In[417]:


train_df,validate_df = train_validate_split(df)


# # Dividing Train Data into Sets of Digits

# In[421]:


sets = {}
for i in range(0,10):
    sets[i] = train_df[train_df['label']==i]


# # Calculation of U for each Set

u = {}
for i in range(0,10):
    get_ipython().run_line_magic('time', 'temp = np.dot(np.array(sets[i])[:,1:].T,np.array(sets[i])[:,1:])')
    temp = np.array(temp)
    print(temp.shape)
    S,U = np.linalg.eigh(temp)
    v = []
    for j in range(0,20):
        ind = S.argmax()
        v.append(np.array(U[:,ind]))
        S[ind] = -99999999
    u[i] = np.array(v).T
    print(u[i].shape)


# # Storing UUT

# In[433]:


svd = {}
for i in range(0,10):
    svd[i] = np.dot(u[i],u[i].T)
#     uu,ss,vv = np.linalg.svd(np.array(sets[i])[:,1:].T, full_matrices=False)
#     uu = uu[:,0:10]
#     print(uu.shape,ss.shape,vv.shape)
#     svd[i] = np.dot(uu,uu.T)


# # Predicting Test Data

# In[434]:


labels = validate_df['label'].values
images = validate_df.drop('label',axis=1).values


# In[450]:


get_ipython().run_cell_magic('time', '', 'pred = []\nfor img in images:\n    min = 99999999999999999999999999\n    ind = 0\n    for i in range(0,10):\n        temp = np.dot(svd[i],img) - img\n        temp = dotProduct(temp,temp)\n        temp = math.sqrt(temp)\n#         temp = np.linalg.norm(temp)\n        if(temp<min):\n            min = temp\n            ind = i\n    pred.append(ind)')


# # Accuracy

# In[451]:


count = 0
for i in range(len(pred)):
    if(pred[i]==labels[i]):
        count += 1
print(count/len(pred))


# In[ ]:

