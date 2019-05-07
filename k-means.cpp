#include <string>
#include <vector>
#include <iostream> 
#include <cmath>
#include <float.h>
#include <math.h>
#include <cuda.h>
#include <sstream> //istringstream
#include <fstream> // ifstream
#include <boost/algorithm/string.hpp>
#include <algorithm>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

using namespace std;
using thrust::placeholders;

class testAndTrainData{
    public:
    double test[10][48000][784];
    double train[10][16000][784];
} ;

// class CSVReader{
//     string fileName;
//     string delimeter;

//     public:
//     CSVReader(string filename, string delm = ",") :
// 			fileName(filename), delimeter(delm)
// 	{ }

//     vector<vector<double> > getData();
// };

vector<vector<double> > getData(string fileName){
    ifstream file(fileName);
	vector<vector<double> > dataList;
	std::string line = "";
    long long count = 0 ;
	while (getline(file, line))
	{   
        try{
            vector<string> vec;
            boost::algorithm::split(vec, line, boost::is_any_of(","));
            vector<double> b;
            std::transform(vec.begin(), vec.end(), back_inserter(b), [](const string & astr){ return stod( astr) ; } ) ;
            dataList.push_back(b);
            if(b[0] == 0) count ++ ;
        }

        catch(exception e){
            continue;
        }
		
	}
	file.close();
	return dataList;
}

vector<double> returnNumberVector(vector<double> &vec){
    vec.erase(vec.begin()+0);
    return vec;
}

testAndTrainData calcData()
{
    // Creating an object of CSVWriter
	string = "apparel-trainval.csv";
    testAndTrainData dataObj;
 
	// Get the data from CSV File
	vector<vector<double> > dataList = getData();

    vector<vector<vector<double> >> data(10);    
    vector<vector<vector<double> >> train_data(10);    
    vector<vector<vector<double> >> test_data(10);    
    
	// Print the content of row by row on screen

	for(vector<double> vec : dataList)
	{   
        if (vec[0] == 0){
            vector<double> temp = returnNumberVector(vec);
            data[0].push_back(temp);
        }

        else if(vec[0] == 1){
            vector<double> temp = returnNumberVector(vec);
            data[1].push_back(temp);
        }

        else if(vec[0] == 2){
            vector<double> temp = returnNumberVector(vec);
            data[2].push_back(temp);
        }

        else if(vec[0] == 3){
            vector<double> temp = returnNumberVector(vec);
            data[3].push_back(temp);
        }
        
        else if(vec[0] == 4){
            vector<double> temp = returnNumberVector(vec);
            data[4].push_back(temp);
        }

        else if(vec[0] == 5){
            vector<double> temp = returnNumberVector(vec);
            data[5].push_back(temp);
        }

        else if(vec[0] == 6){
            vector<double> temp = returnNumberVector(vec);
            data[6].push_back(temp);
        }

        else if(vec[0] == 7){
            vector<double> temp = returnNumberVector(vec);
            data[7].push_back(temp);
        }

        else if(vec[0] == 8){
            vector<double> temp = returnNumberVector(vec);
            data[8].push_back(temp);
        }

        else if(vec[0] == 9){
            vector<double> temp = returnNumberVector(vec);
            data[9].push_back(temp);
        }

	}

    for(int i=0; i<10; i++){
        int split_size = int(data[i].size()*0.8);
        vector<vector<double> > split_lo(data[i].begin(), data[i].begin() + split_size);
        vector<vector<double> > split_hi(data[i].begin() + split_size, data[i].end());
        train_data[i] = split_lo;
        test_data[i] = split_hi;
    }
    
    dataObj.train = train_data;
    dataObj.test = test_data;

	return dataObj;
}

double all_Acc = 0;

__global__
void parallelCalcMean(double *d_sample,double *d_result,int r_size,int c_size,int pitch){

    extern __shared__ double sdata[];
    double x = 0.0;

    double * p = &d_sample[blockIdx.x * r_size];

    for(int i=threadIdx.x; i < c_size; i += blockDim.x) {
        x += p[i];
    }

    sdata[threadIdx.x] = x;
    __syncthreads();

    for(int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if(threadIdx.x < offset) {
            sdata[threadIdx.x] += sdata[threadIdx.x + offset];
        }
        __syncthreads();
    }

    if(threadIdx.x == 0) {
        d_result[blockIdx.x] = sdata[0];
    }
    
}

void convert2DVectorToArray(vector<vector<double> > &matrix, double array[48000][784]){
    for(int i = 0;i<matrix.size();i++){
        for(int j = 0;j<matrix[0].size();j++){
            array[i][j] = matrix[i][j];
        }
    }
}

vector<double> calcMean(vector<vector<double> > &sample){
    float* d_sample;
    size_t pitch;
    double *d_result;       //of size 784
    vector<double> rslt(784,0);
    double array[48000][784];

    cudaMallocPitch((void**)&d_sample, &pitch, 48000 * sizeof(double), 784);
    convert2DVectorToArray(sample,array);
    
    cudaMalloc(&d_result,784);

    int r_size = sample.size();
    int c_size = sample[0].size();

    cudaMemcpy2D(d_sample, pitch, array, 784*sizeof(float), 784*sizeof(float), 48000, cudaMemcpyHostToDevice);
    cudaMalloc(&d_result,784);

    dim3 blocksize(784);
    dim3 gridsize(784);

    float* rslt_array = new float[786]; 

    size_t shmsize = sizeof(double) * (size_t)blocksize.x;
    parallelCalcMean<double><<<gridsize, blocksize, shmsize>>>(d_sample, d_result, r_size, c_size);

    cudaMemcpy(rslt_array, d_result, 786 * sizeof(double), cudaMemcpyDeviceToHost);

    for (int i = 0;i<rslt.size();i++)
        rslt[i] = rslt_array[i];

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
        dist += pow((double)(a[i]-b[i]),2.0);
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
    
    cout <<"Total accuracy "<<all_Acc/tot_length<<endl;
    return 0;
}