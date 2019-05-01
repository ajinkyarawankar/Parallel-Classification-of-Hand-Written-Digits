import numpy as np

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
        print (rslt)
        return rslt
    
    if (len(Ashape) == 1 and len(Bshape) > 1):
        rslt = vectorMulti2(A,B)
        print (rslt)
        return rslt
    
    if (len(Ashape) > 1 and len(Bshape) > 1):
        rslt = matMul(A,B)
        print (rslt)
        return rslt
    
    if (len(Ashape) == 1 and len(Bshape) == 1):
        rslt = oneDProduct(A,B)
        print (rslt)
        return rslt


def main():
    # A = [[1,2,4],[3,4,3],[9,2,1],[2,3,4]]
    A = [1,2,4]

    # A = 2
    # B = [3,4,5]
    B = [[11,12],[13,14],[2,1]]
    A = np.array(A)
    B = np.array(B)

    dotProduct(A,B)

main()
