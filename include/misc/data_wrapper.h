/* 
 * File:   data_wrapper.h
 * Author: ryan
 *
 * Created on 02 April 2017, 14:39
 */

#ifndef DATA_WRAPPER_H
#define DATA_WRAPPER_H

#include "tools/math.h"

#include <Eigen/Dense>
#include <vector>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

using namespace std;
using namespace Eigen;

struct DataWrapper{
    VectorXi training_labels;
    MatrixXf training_images;
    
    VectorXi validation_labels;
    MatrixXf validation_images;
    
    VectorXi testing_labels;
    MatrixXf testing_images;

    DataWrapper(){}  
    void print(){
        cout << "Images: " << endl;
        cout << training_images << "\n";
        cout << "Labels: " << endl;
        cout << training_labels << "\n";
    }
protected:
    void readCSV(const string& path, vector<int>& label, vector< vector<float> >& data){
       ifstream file (path);
       string str;
       while(getline(file,str)){
           istringstream ss(str);
           data.push_back(vector<float>());
           label.push_back(0);
           readLine(ss,label.back(),data.back());
       }
    }  
    void convert2dVecToMat(const vector< vector<float> >& data, MatrixXf& mat){
        mat = MatrixXf(data.size(), data[0].size());
        for (int i = 0; i < data.size(); i++)
            mat.row(i) = VectorXf::Map(&data[i][0],data[i].size());
    }
    void convertVecToVec(const vector<int>& data, VectorXi& mat){
        mat = VectorXi::Map(&data[0],data.size());
    }
private:
    void readLine(istringstream& ss, int& label, vector<float>& data){
       string str;
       getline(ss,str,',');
       label = atoi(str.c_str());
       while(getline(ss,str,',')){
           data.push_back(atof(str.c_str()));
       }
    }

};

struct XORDataWrapper: public DataWrapper{
    XORDataWrapper(const string& training_path, const string& testing_path){
        vector< vector<float> > tmp_training_images;
        vector<int> tmp_training_labels;
        vector< vector<float> > tmp_testing_images;
        vector<int> tmp_testing_labels;
        
        readCSV(training_path,tmp_training_labels,tmp_training_images);
        readCSV(testing_path,tmp_testing_labels,tmp_testing_images);
        
        convert2dVecToMat(tmp_training_images,training_images);
        convertVecToVec(tmp_training_labels,training_labels);
        
        convert2dVecToMat(tmp_testing_images,testing_images);
        convertVecToVec(tmp_testing_labels,testing_labels);  
    }
};

struct MNISTDataWrapper: public DataWrapper{   
    MNISTDataWrapper(const string& training_path, const string& testing_path){
        vector< vector<float> > tmp_training_images;
        vector<int> tmp_training_labels;
        
        vector< vector<float> > tmp_testing_images;
        vector<int> tmp_testing_labels;
        
        readCSV(training_path,tmp_training_labels,tmp_training_images);
        readCSV(testing_path,tmp_testing_labels,tmp_testing_images);
        
        convert2dVecToMat(tmp_training_images,training_images);
        convertVecToVec(tmp_training_labels,training_labels);
        
        convert2dVecToMat(tmp_testing_images,testing_images);
        convertVecToVec(tmp_testing_labels,testing_labels);      
        
        training_images.normalize();
        
//        std::vector<int> indicies(tmp_labels.size());
//        std::iota(indicies.begin(), indicies.end(), 0);
//        const int folds = 10;
//        math::Kfold<vector<int>::const_iterator> kf(folds, indicies.begin(), indicies.end());
//
//        vector<int> train, test;
//        
//        kf.getFold(folds, back_inserter(train), back_inserter(test));
//
//        cout << "Fold " << 1 << " Training Data" << "\n";
//        for(auto x: train){
//            training_labels.push_back(tmp_labels[x]);
//            training_images.push_back(tmp_images[x]);
//            cout << tmp_labels[x] << " ";
//        }
//        cout << "\n";
//        cout << "Fold " << folds << " Testing Data" << "\n";
//        for(auto x: test){
//            validation_labels.push_back(tmp_labels[x]);
//            validation_images.push_back(tmp_images[x]);
//            cout << tmp_labels[x] << " ";
//        }
//        cout << "\n";
    }
};


#endif /* DATA_WRAPPER_H */

