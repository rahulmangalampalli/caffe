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
	boost::shared_ptr<Net<float> > net_;
	int count = 1;
	int tt_count =0;
	vector<vector<float>> v;
	vector<std::string> labels;
	vector<int> totals;
	net_ = boost::shared_ptr<Net<float> >(new Net<float>(argv[1], caffe::TEST));
	net_->CopyTrainedLayersFrom(argv[2]);
    std::string inputDirectory = argv[3];
    DIR *directory = opendir (inputDirectory.c_str());
    struct dirent *_dirent = NULL;
    if(directory == NULL)
    {
        cout << "Cannot open Input Directory"<<endl;
        return 1;
    }
    while((_dirent = readdir(directory)) != NULL)
    {
		if( !strcmp(_dirent->d_name, ".")) continue;
        if( !strcmp(_dirent->d_name, "..")) continue;
        std::string fol = inputDirectory + "/" +std::string(_dirent->d_name);
		DIR *folder = opendir (fol.c_str());
        struct dirent *_dirent_2 = NULL;

		std::string label = std::string(_dirent->d_name); 

		while((_dirent_2 = readdir(folder)) != NULL){
			if( !strcmp(_dirent_2->d_name, ".")) continue;
            if( !strcmp(_dirent_2->d_name, "..")) continue;
			std::string im = fol + "/" +std::string(_dirent_2->d_name);
			Mat image = imread(im.c_str());
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
		tt_count+=1;
		v1.push_back(count);
        v.push_back(v1);
		cout << "Read image:" << im.c_str() << endl;
		}
	   totals.push_back(tt_count);
	   tt_count = 0;
	   count+=1;
	   cout << endl;
	   cout << "Done with label:" << label <<endl; 
	   cout << endl;
	   labels.push_back(label);
	   closedir(folder);	
    }
	std::cout << "Number of Data Points:" << v.size() << std::endl << "Number of Features:" << v[0].size() -1 << endl;
    closedir(directory);

	std::ofstream out("train.csv");

	std::cout << endl;

	std::ofstream output_file("labels.txt");
    std::ostream_iterator<std::string> output_iterator(output_file, "\n");
    std::copy(labels.begin(), labels.end(), output_iterator);

	for (int i = 0; i < count-1; i++)
	{
		cout << "Number of images in label " << labels.at(i) <<" are "<< totals.at(i) << endl;

	}
	
	cout << endl;
	cout << "Saved lables.txt" << endl;
	std::cout << "Saved train.csv" << endl;

	for (auto& row : v) {
  		for (auto col : row)
    		out << col <<',';
  		out << '\n';
		}
	return 0;
}
