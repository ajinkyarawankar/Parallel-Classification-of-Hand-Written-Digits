#include "dotProduct.cpp"
#include "eigen.cpp"
#include "fileRead.h"
using namespace std;



vector<vector<vector<double>>> svd(vector<vector<vector<double>>> sets){
	vector<vector<vector<double>>> u(10);
	Eigen eig;
	for(int i=0;i<10;i++){
		cout<<i<<endl;
		vector<vector<double>> t_set = transpose(sets[i]);
		print_mat(t_set);
		
		vector<vector<double>> temp = dotProduct(t_set,sets[i]);
		print_mat(temp);
		eig = eigen(temp);
		vector<vector<double>> v;  // check for max eigen 
		for(int j=0;j<20;j++){
			v.push_back(eig.eigen_vectors[j]);
		}
		u[i] = v;
		print_mat(v);
	}

	vector<vector<vector<double>>> svd(10);

	for(int i=0;i<10;i++){
		vector<vector<double>> ui_t = transpose(u[i]); 
		svd[i] = dotProduct(u[i],ui_t);
	}

	return svd;
}


double prediction(vector<vector<vector<double>>> &svd,vector<vector<vector<double>>> &test){
	vector <vector<int>> pred(10);

	for(int set = 0;set<10;set++){
		for(int i=0;i<test[set].size();i++){
			double min = 99999999;
			long long ind = 0;

			for(int j=0;j<10;j++){

				
				vector<double> temp = dotProduct(svd[j],test[set][i]);
				for(int k=0;k<temp.size();k++){
					temp[k] = temp[k] - test[set][i][k];
				}
				double temp1 = dotProduct(temp,temp);
				temp1 = sqrt(temp1);

				if(temp1<min){
					min = temp1;
					ind = j; 
				}
			}
			pred[set].push_back(ind);
		}
	}

	double correct = 0;
	double total  = 0;
	double accuracy = 0;
	for(int set=0;set<10;set++){
		for(int j=0;j<pred[set].size();j++){
			if(pred[set][j] == set)
				correct+=1;
			total+=1;
		}
	}

	accuracy = correct/total;
	return accuracy;

}

int main(){
	testAndTrainData data;
	data = calcData();
	cout<< data.train.size()<<endl;
	cout<< data.train[0].size()<<endl;
	cout<< data.train[0][0].size()<<endl;
	
	vector<vector<vector<double>>> sv = svd(data.train);
	cout<<prediction(sv,data.test);
	return 0;
}