#include <iostream>
#include <ctime>
#include <cstdlib>
#include <vector>
#include <math.h>
using namespace std;


bool checkDiagonal(vector<vector<double>> &arr){
    for(int i=0;i<arr.size();i++){
        for(int j=0;j<arr[i].size();j++){
            if (i == j)
                continue;
            else{
                if (abs(arr[i][j]) > 0.001)
                    return 0;
            }
         }
    }
    return 1;
}

// int order = 4;
// int p = order - 1;
// int q = order - 1;

vector<double> Eigen_values;
vector<vector<double>> eigenvectors_list;


double norm(vector<double> x){
	double result = 0;
	for(int i=0;i<x.size();i++){
		result += x[i]*x[i];
	}
	result = sqrt(result);
	return result;
}

vector<vector<double> > vect_dot(vector<double> x,vector<double> y,int index){
	vector<vector<double> > result(x.size());
	for(int i=0;i<x.size();i++){
		result[i] = vector<double> (x.size(),0.0);
		//optimization
		// start loop at index position because previous are all 0's
		// end loop at index+p because further elements are all 0's
		for(int j=index;j<x.size();j++){
			if(x[i]!=0)
			result[i][j] = x[i] * y[j];
		}
	}
	return result;
}

vector<vector<double> > identity(int order){
	vector<vector<double> > I(order);
	for(int i=0;i<order;i++){
		I[i] = vector<double> (order);
		for(int j=0;j<order;j++){
			if(i==j)
				I[i][j] = 1;
			else
				I[i][j] = 0;
		}
	}
	return I;
}

vector<vector<double> > mat_dot(vector<vector<double> > x,vector<vector<double> > y){
	vector<vector<double> > result(x.size());
	for(int i=0;i<x.size();i++){
		result[i] = vector<double> (x.size(),0.0);
		for(int j=0;j<y[0].size();j++){
			result[i][j] = 0;
			for(int k=0;k<y.size();k++){
				result[i][j] += x[i][k]*y[k][j];
			}
		}
	}
	return result;
}


vector<vector<double> > transpose(vector<vector<double> > x){
	vector<vector<double> > result(x[0].size(),vector<double>(x.size(),0));
	for(int i=0;i<x.size();i++){
		for(int j=0;j<x[0].size();j++){
			result[j][i] = x[i][j];
			
		}
	}
	return result;
}

vector<vector<double> > zeros(vector<vector<double>> x){
	vector<vector<double> > result(x.size(),vector<double>(x[0].size(),0));
	for(int i=0;i<x.size();i++){
		for(int j=0;j<x[0].size();j++){
			if(abs(x[i][j]) <0.0000001)
				result[i][j]=0;
			else
				result[i][j] = x[i][j];
			
		}
	}
	return result;
}


vector<double> mat_vect(vector<vector<double> > x,vector<double> y){
	vector<double> result(x.size());
	for(int i=0;i<x.size();i++){
		result[i]=0;
		for(int j=0;j<y.size();j++){
			result[i] += x[i][j] * y[j];
		}
	}
	return result;
}


void print_vector(vector<double> x){
	for(int i=0;i<x.size();i++){
		cout<<x[i]<<" ";
	}
	cout<<endl;
}

void print_mat(vector<vector<double> > x){
	for(int i=0;i<x.size();i++){
		for(int j=0;j<x.size();j++){
			cout<<x[i][j]<<" ";
		}
		cout<<endl;
	}
}

vector<vector<double> > house(vector<double> x,int index){
	vector<double> u;
	vector<double> e1;
	for (int i=0;i<x.size();i++){
		u.push_back(x[i]); 
		e1.push_back(0);
	}
	for(int i=0;i<index;i++){
		u[i]=0;
	}
	double n = norm(u);
	u[index] = u[index] - n;
	n = norm(u);
	if(n!=0)
	//optimization
	for(int i=index;i<x.size();i++){
		u[i]=u[i]/n;
	}	
	vector<vector<double> > I = identity(x.size());
	vector<vector<double> > v = vect_dot(u,u,index);
	for(int i=0;i<x.size();i++){
		for(int j=0;j<x.size();j++){
			v[i][j] = I[i][j] - 2*v[i][j];
		}
	}
	return v;
}

// vector<double> backward(vector<vector<double> > &U, vector<double> t){
// 	vector<double> x(order);
// 	x[order-1] = t[order-1]/U[order-1][order-1];
// 	for(int i=order-2;i>=0;i--){
// 		double temp = t[i];
// 		for(int j=order-1;j>i;j--){
// 			temp -= U[i][j]*x[j];
// 		}
// 		if(U[i][i] == 0)
// 			x[i]=0;
// 		else
// 			x[i] = temp/U[i][i];
// 	}
// 	return x;
// }

vector<vector<vector<double>>> qr(vector<vector<double>> A){
	vector<vector<vector<double>>> res;
	int order = A.size();
	// p = A.size() - 1;
	// q = A[0].size() - 1;
	vector<double> x(order);
	vector<vector<double> > P = A;
	vector<vector<double> > Q = identity(order);
	vector<vector<double> > H;
	for(int i=0;i<order-1;i++){
		for(int j=0;j<order;j++){
			x[j] = P[j][i];
		}
		H = house(x,i);
		P = mat_dot(H,P);
		Q = mat_dot(H,Q);	
	}
	vector<vector<double> > R = mat_dot(Q,A);
	
	Q = transpose(Q);
	Q = zeros(Q);
	R = zeros(R);
	res.push_back(Q);
	res.push_back(R);
	return res;

}

vector<vector<double>> qrFactorization(vector<vector<double>> &arr){
    vector<vector<double>> temp = arr;
    vector<vector<double>> Q,R;
    int i = 0;
    vector<vector<double>> eigenvectors = identity(arr.size());
    while(1){
    	vector<vector<vector<double>>> res = qr(temp);
        Q = res[0];
        R = res[1];
        temp = mat_dot(R,Q);
        // cout<<"temp"<<endl;
        // print_mat(Q);
        
        eigenvectors = mat_dot(eigenvectors, Q);
        // break;
        // cout<<i<<endl;
        if(checkDiagonal(temp)){
        	// cout<<"b";
            cout<<"Number of Factorizations: "<<i+1;
            break;
        }
        else
            i += 1;
    }
    eigenvectors = transpose(eigenvectors);
    for(int i=0;i<arr.size();i++)
        eigenvectors_list.push_back(eigenvectors[i]);
    return temp;

}

void printLambda(vector<vector<double>> &arr){
	int count = 1;
	for (int i=0;i<arr.size();i++){
		for(int j=0;j<arr[0].size();j++){
			if(i == j){
				double temp = arr[i][j];
				if(abs(temp) < 0.000000000001)
					temp = 0;
				Eigen_values.push_back(temp);
				// print("Lamda"+str(count) +": " + str(temp))
				count += 1;
			}
		}
	}
}


int main(){
	vector<vector<double> > A{ { 1, 3, 1, 9, 1, 2 }, { 3, 2, 1, 0, 1, 2 }, { 1, 1, 3, 4, 1, 2 }, { 9, 0, 4, 1, 1, 2 }, { 1, 1, 1, 1, 1, 2 }, { 2, 2, 2, 2, 2, 2 } };
	vector<vector<vector<double>>> res = qr(A);
	cout<<"Q-\n";
	print_mat(res[0]);
	cout<<"R-\n";
	print_mat(res[1]);
	vector<vector<double>> arr = qrFactorization(A);
	printLambda(arr);
	cout<<endl;
	cout<<"Eigen Vectors are"<<endl;
	print_mat(eigenvectors_list);
	cout<<"Eigen Values are"<<endl;
	print_vector(Eigen_values);

	return 0;
	
}
