#include <iostream>
#include <stdio.h>  // for snprintf
#include <string>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml.hpp>
#include <caffe/caffe.hpp>
#include "dirent.h"

using namespace cv::ml;
using namespace std;
using namespace cv;
using namespace caffe;

int main(int argc, char* argv[])
{
	cout << "Training the model" << endl;
  	Ptr<TrainData> TrainData = TrainData::loadFromCSV(argv[1],1,-1,-1);
	cv::Mat matResults(0,0,cv::DataType<int>::type);
	TrainData->setTrainTestSplitRatio(0.8,true);  	
	Ptr<KNearest> model = KNearest::create();
	model->setDefaultK(3);
	model->train(TrainData);
	model->save("train.knn");
	cout<<"Training done on :"<< TrainData->getNTrainSamples() <<endl;
	cout<<endl;	
	cout<<"Accuracy on Test Set of size "<< TrainData->getNTestSamples() <<" datapoints is:" << model->calcError(TrainData, true, matResults)<<endl; 
	cout<<endl;	
	return 0;
}

