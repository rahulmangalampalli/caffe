#include <vector> // to get vector utility
#include <opencv2/core/core.hpp>  // for opencv utilities
#include <string>   // string library which support usage of strings
#include <opencv2/highgui/highgui.hpp> //to get gui related functions from opencv
#include <opencv2/imgproc/imgproc.hpp> //to get image processing functions from opencv
#include <caffe/caffe.hpp> // to deal with caffe basic operations

using namespace std;
using namespace cv;
using namespace caffe;


void infer(string prototxt, string caffemodel, string data_path)
{

    boost::shared_ptr<Net<float> > net_; // intialize net object for reading prototxt and caffemodel
    vector<vector<float>> v;
    net_ = boost::shared_ptr<Net<float> >(new Net<float>(prototxt, caffe::TEST)); //read prototxt 
    net_->CopyTrainedLayersFrom(caffemodel); // read caffemodel
    Mat image = imread(data_path); //read image
    Mat img; // Fore storing resize output
    resize(image, img, Size(64, 64)); //resize image
    img.convertTo(img,CV_32F); // Type cast input image to float
    img /= 255.0; 	
    // Load image into buffers for inferencing

    std::vector<cv::Mat> input_channels; // initialize input blob for loading image
    Blob<float>* input_layer1 = net_->input_blobs()[0];  // get input blob object for size intialization
    int width = input_layer1->width(); // get blob height
    int height = input_layer1->height(); // get blob height
    int num_channels_ = input_layer1->channels(); // get blob width
    float* input_data = input_layer1->mutable_cpu_data(); 
    	for (int i = 0; i < input_layer1->channels(); ++i) { //push data into the blob channel wise
     		cv::Mat channel(height, width, CV_32FC1, input_data);
     		input_channels.push_back(channel);
     		input_data += width * height;
    	}

    cv::split(img, input_channels);
   
    CHECK(reinterpret_cast<float*>(input_channels.at(0).data)
        == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";

    net_->Forward(); //Inference starts

    const float* embeddings = NULL; // Initialize embedding vector for storing class probability
    embeddings = net_->blob_by_name("dense_5/dense_5/kernel/sig")->cpu_data(); //Collect last layer output into embedding vector
    float val = *embeddings; // get the fire probability into a variable
    std::string label = "Fire Probability: " + std::to_string(val*100.0); // Output probability on input image
    cv::putText(image,label,cv::Point(10, 25),cv::FONT_HERSHEY_SIMPLEX,0.7,CV_RGB(0, 255, 0),2); 
    cv::imshow("Detection", image); // shows probability on input image
    cv::waitKey(); // Press any key to quit
 }


void infer_vid(string prototxt, string caffemodel, string data_path){

    boost::shared_ptr<Net<float> > net_;
    vector<vector<float>> v;
    net_ = boost::shared_ptr<Net<float> >(new Net<float>(prototxt, caffe::TEST)); //read prototxt 
    net_->CopyTrainedLayersFrom(caffemodel); // read caffemodel

    
    cv::VideoCapture cap(data_path);
    if (!cap.isOpened())
    {
        std::cout << "!!! Failed to open file: " << data_path << std::endl;
        return;
    }

    cv::Mat frame;
    for(;;)
    {

       if (!cap.read(frame))             
           break;
       Mat img;
       resize(frame, img, Size(64, 64)); //resize image
       img.convertTo(img,CV_32F);
       img /= 255.0;
       
       std::vector<cv::Mat> input_channels;
       Blob<float>* input_layer1 = net_->input_blobs()[0];
       int width = input_layer1->width();
       int height = input_layer1->height();
       int num_channels_ = input_layer1->channels();
       float* input_data = input_layer1->mutable_cpu_data();
       for (int i = 0; i < input_layer1->channels(); ++i) {
           cv::Mat channel(height, width, CV_32FC1, input_data);
           input_channels.push_back(channel);
           input_data += width * height;
    	}

    	cv::split(img, input_channels);
   
    	CHECK(reinterpret_cast<float*>(input_channels.at(0).data)
        == net_->input_blobs()[0]->cpu_data())
    	<< "Input channels are not wrapping the input layer of the network.";

    	net_->Forward(); //Inference starts

    	const float* embeddings = NULL;
    	embeddings = net_->blob_by_name("dense_5/dense_5/kernel/sig")->cpu_data(); //Collect last layer output
    	float val = *embeddings;
   	std::string label = "Fire Probability: " + std::to_string(val*100.0);
    	cv::putText(frame,label,cv::Point(10, 25),cv::FONT_HERSHEY_SIMPLEX,0.7,CV_RGB(0, 255, 0),2);
        cv::imshow("Detection", frame);
        // Press  ESC on keyboard to exit
        char c=(char)waitKey(25);
        if(c==27)
            break;

    }

}


int main(int argc, char **argv)
{
	string model_file = argv[3];
	string weights_file = argv[2];
	Caffe::set_mode(Caffe::CPU);
	string im_names=argv[1];
        int options;
        cout << "Please input 0 if you have provided video file else 1:"<<endl;
        cin >> options;
        if(options == 1)
            infer(model_file,weights_file,im_names);
        else
            infer_vid(model_file,weights_file,im_names);
        return 0;
}
