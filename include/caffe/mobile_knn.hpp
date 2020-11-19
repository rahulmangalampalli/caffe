#include <iostream>
#include <stdio.h> 
#include <string>



using namespace std;   // stdout library for printing values 

class Classify{
public:
    Classify();
    ~Classify();
    int create_csv(string prototxt, string caffemodel, string data_path);
    int train_csv(string csv_file);
    int infer(string prototxt, string caffemodel, string data_path, string knn_path, string labels);
};