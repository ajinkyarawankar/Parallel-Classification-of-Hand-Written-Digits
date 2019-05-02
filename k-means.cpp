#include <string>
#include <vector>
#include <iostream> 
#include <cmath>
#include<float.h>
#include<math.h>
// #include "dotProduct.cpp"
#include "fileRead.h"

using namespace std;

double all_Acc = 0;
void print(vector<double> &vec){
    for(int i=0;i<vec.size();i++){
        cout<<vec[i]<<" ";
    }
    cout<<endl;
}

vector<double> calcMean(vector<vector<double> > &sample){
    vector<double> rslt(784,0);

    for (int i =0 ; i<sample[0].size(); i++){
        int total = 0;
        for (int j =0; j<sample.size(); j++){
            total += sample[j][i];
        }
        rslt[i] = total;
    }   

    for(int i =0 ;i<sample.size();i++){
        rslt[i] = rslt[i]/sample.size();
    }

    // print(rslt);
    return rslt;
}

vector<vector<double> > findMeanOfData(vector<vector<vector<double>> > &train_data){
    vector<vector<double> > temp;
    vector<vector<double> > result(10);
    
    for (int i =0 ;i<10;i++){
        temp = train_data[i];
        result[i] = calcMean(temp);
    }
    return result;
}

double euclideanDistance(vector<double> &a, vector<double> &b){
    double dist = 0;
    for (int i=0;i<a.size();i++){
        dist += pow((a[i]-b[i]),2);
    }

    dist = sqrt(dist);
    return dist;
}

vector<double> prediction(vector<vector<double >> &test_sample, vector<vector<double >> &mean){

    vector<double> pred;

        for(int j=0;j<test_sample.size();j++){

            double min = DBL_MAX;
            int pred_class = 0;
            double dist = 0;

            for(int k=0;k<mean.size();k++){

                dist = euclideanDistance(mean[k],test_sample[j]);
                if(dist<min){
                    min = dist;
                    pred_class = k;
                }
            }

            pred.push_back(pred_class);
        }
    
    return pred;
}

double accuracy(vector<double > &pred,int digit_class){
    double total_acc = 0;
    double size = 0;
    double acc = 0;
    
    for(int i=0;i<pred.size();i++){

        if(pred[i] == digit_class){
            total_acc += 1;
            all_Acc += 1;
        }
            
    }

    acc = total_acc/pred.size();
    cout<<"Accuracy of Set "<<digit_class<<" is "<<acc<<endl;
    
    return total_acc;
}

int main(){
    int tot_length = 0 ;
    testAndTrainData obj = calcData();
    // cout<<sizeof(obj);
    vector<vector<vector<double> >> train_data = obj.train;
    vector<vector<vector<double> >> test_data = obj.test;

    vector<vector<double> > mean = findMeanOfData(train_data);

    for (int i=0;i<test_data.size();i++){
        tot_length += test_data[i].size();
    }

    for (int i=0;i<test_data.size();i++){
        vector<double> pred = prediction(test_data[i],mean);
        double acc = accuracy(pred,i);
    }
    // vector<double> pred = prediction(test_data,mean);

    // for (int i=0;i<10;i++)
    //     print(pred[i]);

    // accuracy(pred);
    cout <<"Total accuracy "<<all_Acc/tot_length<<endl;
    return 0;
}