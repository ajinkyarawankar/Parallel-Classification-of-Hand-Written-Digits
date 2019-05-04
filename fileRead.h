#ifndef FILEREAD_H
#define FILEREAD_H

#include <string>
#include <vector>
#include <sstream> //istringstream
#include <iostream> // cout
#include <fstream> // ifstream
// #include <boost/python.hpp>
#include <boost/algorithm/string.hpp>
#include <algorithm>

using namespace std;
// using namespace boost::python;

class testAndTrainData{
    public:
    vector<vector<vector<double> >> test;
    vector<vector<vector<double> >> train;
} ;

class CSVReader{
    string fileName;
    string delimeter;

    public:
    CSVReader(string filename, string delm = ",") :
			fileName(filename), delimeter(delm)
	{ }

    vector<vector<double> > getData();
};

vector<vector<double> > CSVReader::getData(){
    ifstream file(fileName);
 
	vector<vector<double> > dataList;
 
	std::string line = "";
    long long count = 0 ;
	// Iterate through each line and split the content using delimeter
	while (getline(file, line))
	{   
        try{
            vector<string> vec;
            boost::algorithm::split(vec, line, boost::is_any_of(delimeter));
            vector<double> b;
            std::transform(vec.begin(), vec.end(), back_inserter(b), [](const string & astr){ return stod( astr) ; } ) ;
            dataList.push_back(b);
            if(b[0] == 0) count ++ ;
        }

        catch(exception e){
            // cout << "Error "<<endl;
            continue;
        }
		
	}
	// Close the File
	file.close();
    // cout << "Total zeros "<<count<<endl;
	return dataList;
}

vector<double> returnNumberVector(vector<double> &vec){
    vec.erase(vec.begin()+0);
    return vec;
}

testAndTrainData calcData()
{
    // Creating an object of CSVWriter
	CSVReader reader("apparel-trainval.csv");
    testAndTrainData dataObj;
 
	// Get the data from CSV File
	vector<vector<double> > dataList = reader.getData();

    vector<vector<vector<double> >> data(10);    
    vector<vector<vector<double> >> train_data(10);    
    vector<vector<vector<double> >> test_data(10);    
    
	// Print the content of row by row on screen

	for(vector<double> vec : dataList)
	{   
        if (vec[0] == 0){
            vector<double> temp = returnNumberVector(vec);
            // zeros.push_back(temp);
            data[0].push_back(temp);
        }

        else if(vec[0] == 1){
            vector<double> temp = returnNumberVector(vec);
            // ones.push_back(temp);
            data[1].push_back(temp);
        }

        else if(vec[0] == 2){
            vector<double> temp = returnNumberVector(vec);
            // twos.push_back(temp);
            data[2].push_back(temp);
        }

        else if(vec[0] == 3){
            vector<double> temp = returnNumberVector(vec);
            // threes.push_back(temp);
            data[3].push_back(temp);
        }
        
        else if(vec[0] == 4){
            vector<double> temp = returnNumberVector(vec);
            // fours.push_back(temp);
            data[4].push_back(temp);
        }

        else if(vec[0] == 5){
            vector<double> temp = returnNumberVector(vec);
            // fives.push_back(temp);
            data[5].push_back(temp);
        }

        else if(vec[0] == 6){
            vector<double> temp = returnNumberVector(vec);
            // sixes.push_back(temp);
            data[6].push_back(temp);
        }

        else if(vec[0] == 7){
            vector<double> temp = returnNumberVector(vec);
            // sevens.push_back(temp);
            data[7].push_back(temp);
        }

        else if(vec[0] == 8){
            vector<double> temp = returnNumberVector(vec);
            // eights.push_back(temp);
            data[8].push_back(temp);
        }

        else if(vec[0] == 9){
            vector<double> temp = returnNumberVector(vec);
            // nines.push_back(temp);
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
    // for(int i=0; i<10; i++){
    //     cout <<i<<"original"<<data[i].size()<<endl;
    //     cout <<i<<"train"<<train_data[i].size()<<endl;
    //     cout <<i<<"test"<<test_data[i].size()<<endl;
    // }
    
    dataObj.train = train_data;
    dataObj.test = test_data;

	return dataObj;
}

// int main(){
//     return 0;
// }

#endif