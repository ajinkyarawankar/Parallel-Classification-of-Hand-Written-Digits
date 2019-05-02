import copy
import numpy as np

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
