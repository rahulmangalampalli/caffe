#include "include/caffe/mobile_knn.hpp"

int main(int argc, char** argv)
{
  /* Check if command-line arguments are correct */
  if(argc !=4)
  {
     cout<<"\nWrong number of arguments, please enter 3 arguments 1. csv file name with .prototxt extension 2. .caffemodel extension 3. .csv extenstion";
     exit(0);
  }
  else {
  /* Reading arguments from command line */
  string proto = argv[1];
  string caffem = argv[2];
  string data = argv[3];
  /* Creating object for Linearreg class */
  Classify cl;
  /* calling linear_train function to train model on given csv file, col1 and col2 */
  cl.create_csv(proto,caffem,data);
  string ch;
  cout << "Created csv file and saved in the present working directory.Do you want to train it (y/n)?";
  cin >> ch;
  if(ch == "y")
    cl.train_csv("train.csv");
  
  else if(ch == "n")
  cout << "No file trained for now"<<endl;

  else{
  cout<<"Wrong input.Hence exiting"<<endl;
  return 0;
  }
  string path;
  cout << "Provide path for inferencing any image:";
  cin >> path;

  cl.infer(proto,caffem,path,"train.knn","labels.txt");

  return 0;
}

return 0;
}

