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


int main( int argc, char* argv[]) {

    boost::shared_ptr<Net<float> > net_;
	vector<vector<float>> v;
    net_ = boost::shared_ptr<Net<float> >(new Net<float>(argv[1], caffe::TEST));
	net_->CopyTrainedLayersFrom(argv[2]);
	Mat image = imread(argv[3]);
	Mat img;
	resize(image, img, Size(224, 224));
   	Blob<float>* input_layer = net_->input_blobs()[0];
    int num_channels_ = input_layer->channels();
    cv::Size input_geometry_;
    input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
    input_layer->Reshape(1, num_channels_, input_geometry_.height, input_geometry_.width);
    net_->Reshape();
    std::vector<cv::Mat> input_channels;
    Blob<float>* input_layer1 = net_->input_blobs()[0];
    int width = input_layer1->width();
    int height = input_layer1->height();
    float* input_data = input_layer1->mutable_cpu_data();
    	for (int i = 0; i < input_layer1->channels(); ++i) {
     		cv::Mat channel(height, width, CV_32FC1, input_data);
     		input_channels.push_back(channel);
     		input_data += width * height;
    	}

    	cv::Mat sample_float;
   		img.convertTo(sample_float, CV_32FC3);
    	cv::split(sample_float, input_channels);
   
    CHECK(reinterpret_cast<float*>(input_channels.at(0).data)
        == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";

    net_->Forward();

	const float* embeddings = NULL;
    embeddings = net_->blob_by_name("MobilenetV1/Logits")->cpu_data();
        
	vector<float> v1;

	for(int i = 0; i < 1001; i++ ){
		v1.push_back(*embeddings);
		embeddings++;
	}

	v.push_back(v1);// data is in format of rows x col but classifier shape is col x rows

	cv::Mat data(0, 1, cv::DataType<float>::type); // Hence till line 65, the code reshapes data to col x row

    for (unsigned int i = 0; i < v.size(); ++i)
	{
  		cv::Mat Sample(1, v[0].size(), cv::DataType<float>::type, v[i].data());
  		data.push_back(Sample);
	}

	cv::Mat result;
	
	Ptr<KNearest> knn_classify = Algorithm::load<KNearest>(argv[4]);
    knn_classify-> findNearest(data,knn_classify->getDefaultK(),result);

	std::ifstream in("labels.txt");

	std::string str;

	vector<std::string> names;

	while (std::getline(in, str))
	{
    	if(str.size() > 0)
        	names.push_back(str);
	}

    cout << "The class of this image is:" << names.at((int)result.at<float>(0)-1) << endl;;
    return 0;

}