#include <string>
#include <vector>
#include <iostream> // cout
#include <typeinfo>

using namespace std;

vector<double> dotProduct(vector<vector<double> > &A, vector<double> &B){
    vector<double> rslt(A.size());
    try{    
        for (int i=0; i<A.size(); i++){
            for (int j=0; j<A[0].size(); j++){
                rslt[i] += A[i][j] * B[j];
            }
        }
    }

    catch(exception e){
        cout << "Dimensions incompatible!"<<endl;
    }
        
    return rslt;
}

vector<double> dotProduct(vector<double> &A, vector<vector<double> > &B){
    vector<double> rslt(B[0].size(),0);
    try{
        for (int i=0; i<B[0].size(); i++){
            for (int j=0; j<A.size(); j++){
                rslt[i] += A[j] * B[j][i];
            } 
        }
    }

    catch(exception e){
        cout << "Dimensions incompatible!"<<endl;
    }

    return rslt;   
}

vector<vector<double> > dotProduct(vector<vector<double> > &A, vector<vector<double> > &B){
    vector<vector<double> > rslt(A.size(),vector<double> (B[0].size(),0));
    try{
        for(int i=0; i<A.size(); i++){
            for(int j=0; j<B[0].size(); j++){
                for(int k=0; k<A[0].size(); k++){
                    rslt[i][j] += A[i][k] * B[k][j];
                }
            }
        }
    }

    catch(exception e){
        cout << "Dimensions incompatible!"<<endl;
    }

    return rslt;
}

double dotProduct(vector<double> &A , vector<double> &B){
    double rslt = 0;
    try{
        for(int i=0; i<A.size(); i++){
            rslt += A[i] * B[i];
        }
    }
    catch(exception e){
        cout << "Dimensions incompatible!"<<endl;
    }

    return rslt;
}

int main(){
    // vector<double> A = {1,2,4};
    vector<vector<double> > A{
        {1,2,4},
        {3,4,3},
        {9,2,1},
        {2,3,4}
    };

    // vector<double> B = {3,4,5};
    vector<vector<double> > B{
        {11,12},
        {13,14},
        {2,1}
    };

    auto a = dotProduct(A,B);
    cout << typeid(a).name() <<endl;
    for (int i =0 ; i<a.size();i++){
        for (int j=0 ; j<a[0].size();j++){
            cout << a[i][j] <<" ";
        }
        cout<<endl;
    }

    return 0;
}