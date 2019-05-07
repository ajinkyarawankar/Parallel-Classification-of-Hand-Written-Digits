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
#include <sys/time.h>
#include <time.h>
#include <ctime>
#include <ratio>
#include <chrono>
using namespace std;
	using namespace std::chrono;
int total_time = 0;
int num_threads_used = 784/784;
class testAndTrainData{
    public:
	vector<vector<vector<double> >> test;
	vector<vector<vector<double> >> train;
} ;

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
	string fileName = "apparel-trainval.csv";
    	testAndTrainData dataObj;
 
	// Get the data from CSV File
	vector<vector<double> > dataList = getData(fileName);

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
    	int rows_in_train = 100;
    	int rows_in_test = 20;
        vector<vector<double> > split_lo(data[i].begin(), data[i].begin() + rows_in_train);
        vector<vector<double> > split_hi(data[i].begin() + rows_in_train, data[i].begin()+(rows_in_train+rows_in_test));
        train_data[i] = split_lo;
        test_data[i] = split_hi;
    }
    
    dataObj.train = train_data;
    dataObj.test = test_data;

	return dataObj;
}

double all_Acc = 0;

__global__
void parallelCalcMean(double *d_sample,double *d_result,int r_size,int c_size,int num_threads){
	unsigned idx = threadIdx.x + blockDim.x*blockIdx.x;

	for(int ind = idx*(784/num_threads);ind<idx*(784/num_threads)+(784/num_threads);ind++){
		int index = ind;

    	if(index<c_size){   
    		double my_result = (double)0;
    		for(int i = 0;i<r_size;i++)
    			my_result += d_sample[(i*c_size)+index];
    		d_result[index] = my_result;
    		//index += gridDim.x * blockDim.x;
    		
    	}
	}

}

void convert2DVectorToArray(vector<vector<double> > &matrix, double array[100][784]){
    for(int i = 0;i<matrix.size();i++){
        for(int j = 0;j<matrix[0].size();j++){
            array[i][j] = matrix[i][j];
        }
    }
}

vector<double> calcMean(vector<vector<double> > &sample){

    int r_size = sample.size();
    int c_size = sample[0].size();
    //int num_threads_used = 784/2;

    double* d_sample;
    double *d_result;       //of size 784
    double* rslt_array = new double[c_size];

    vector<double> rslt(c_size,0);
    double array[100][784];

    const size_t a_size = sizeof(double) * size_t(r_size*c_size);

    convert2DVectorToArray(sample,array);

    cudaMalloc((void**)&d_sample,a_size);
    cudaMalloc(&d_result,sizeof(double)*c_size);

    cudaMemcpy(d_sample,array,a_size,cudaMemcpyHostToDevice);
    auto start = high_resolution_clock::now();
    parallelCalcMean<<<1, num_threads_used>>>(d_sample, d_result, r_size, c_size, num_threads_used);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop-start);
    //cout<<duration.count()<<endl;
    total_time += duration.count();
    cudaMemcpy(rslt_array, d_result, c_size * sizeof(double), cudaMemcpyDeviceToHost);
   
    for (int i = 0;i<rslt.size();i++)
        rslt[i] = rslt_array[i]/r_size;

    cudaDeviceReset();
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

__global__
void parallelCalArrayC(double *a, double *b, double *c, int array_size){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(array_size > idx){
		c[idx] = (a[idx] - b[idx]) * (a[idx] - b[idx]);
		
	}
}

__global__
void parallelEuclidean(double *c,double *rslt, int size){
	extern __shared__ double sdata[];

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	sdata[tid] = 0;

	if(i<size)
		sdata[tid] = c[i];
	__syncthreads();

	for(unsigned int s=1; s<blockDim.x; s*=2){
		if(tid % (2*s) == 0){
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}

	if(tid == 0) rslt[blockIdx.x] = sdata[0];

}

void convert1DvectorToArray(vector<double> &vec,double *array){
	for(int i=0;i<vec.size();i++){
		array[i] = vec[i];
	}
}


double euclideanDistance(vector<double> &a, vector<double> &b){
    	double dist = 0;

    	int size_a = a.size();
    	int size_b = b.size();

    	double *h_a = (double*)malloc(size_a * sizeof(double));
    	double *h_b = (double*)malloc(size_b * sizeof(double));
    	double *distance = (double*)malloc(size_a * sizeof(double));

    	convert1DvectorToArray(a,h_a);
    	convert1DvectorToArray(b,h_b);
   
    	double *d_a;
    	double *d_b;
    	double *d_c;
    	double *d_d;
    	cudaMalloc(&d_a,sizeof(double)*size_a);
    	cudaMalloc(&d_b,sizeof(double)*size_b);
    	cudaMalloc(&d_c,sizeof(double)*size_b);
    	cudaMalloc(&d_d,sizeof(double));

    	cudaMemcpy(d_a,h_a,sizeof(double)*size_a,cudaMemcpyHostToDevice);
    	cudaMemcpy(d_b,h_b,sizeof(double)*size_b,cudaMemcpyHostToDevice);

	auto start = high_resolution_clock::now();
    	parallelCalArrayC<<<1,size_a>>>(d_a, d_b, d_c, size_a);

    	auto stop = high_resolution_clock::now();
    	auto duration = duration_cast<microseconds>(stop-start);

   	//cout<<duration.count()<<endl;

   	total_time += duration.count();
        cudaMemcpy(distance, d_c, sizeof(double)*size_a, cudaMemcpyDeviceToHost);

    	int size = 784;
  	int threadsPerBlock = 1024;
  	int totalBlocks = (size+(threadsPerBlock-1))/threadsPerBlock;
	start = high_resolution_clock::now();
     	parallelEuclidean<<<totalBlocks, threadsPerBlock, threadsPerBlock*sizeof(double)>>>(d_c, d_d, size_a);


     	stop = high_resolution_clock::now();
     	duration = duration_cast<microseconds>(stop-start);

   	//cout<<duration.count()<<endl;

  	total_time += duration.count();

       
    	double result;
	cudaMemcpy(&result, d_d, sizeof(double), cudaMemcpyDeviceToHost);
	cudaDeviceReset();	
    	dist = sqrt(result);
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
    cout<<"Total time - "<<total_time<<"----"<<"No of Threads - "<<num_threads_used<<endl;
    return 0;
}
