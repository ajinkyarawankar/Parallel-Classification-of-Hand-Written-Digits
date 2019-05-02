#include <iostream>
#include <ctime>
#include <cstdlib>
#include <vector>
#include <math.h>
using namespace std;

int order = 4;
int p = order - 1;
int q = order - 1;

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
		for(int j=index;j<x.size() && j<=index+p;j++){
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
	vector<vector<double> > result(order);
	for(int i=0;i<order;i++){
		result[i] = vector<double> (order,0.0);
		for(int j=0;j<order;j++){
			result[i][j] = 0;
			for(int k=0;k<order;k++){
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
	vector<double> result(order);
	for(int i=0;i<order;i++){
		result[i]=0;
		for(int j=0;j<order;j++){
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
	for(int i=0;i<order;i++){
		for(int j=0;j<order;j++){
			cout<<x[i][j]<<" ";
		}
		cout<<endl;
	}
}

vector<vector<double> > house(vector<double> x,int index){
	vector<double> u;
	vector<double> e1;
	for (int i=0;i<order;i++){
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
	for(int i=index;i<order && i<=index+p;i++){
		u[i]=u[i]/n;
	}	
	vector<vector<double> > I = identity(order);
	vector<vector<double> > v = vect_dot(u,u,index);
	for(int i=0;i<order;i++){
		for(int j=0;j<order;j++){
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


int main(){
	vector<vector<double> > A{ { 1, 3, 1, 9 }, { 3, 2, 1, 0 }};
	order = A.size();
	p = A.size() - 1;
	q = A[0].size() - 1;
	vector<double> x(order);
	vector<vector<double> > P =A;
	vector<vector<double> > Q =identity(order);
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
	cout<<"Q-\n";
	print_mat(Q);
	cout<<"R-\n";
	print_mat(R);
	// print_mat(Q);
	
}
