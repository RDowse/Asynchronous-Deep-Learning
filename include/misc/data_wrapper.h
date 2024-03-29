/* 
 * File:   data_wrapper.h
 * Author: ryan
 *
 * Created on 02 April 2017, 14:39
 */

#ifndef DATA_WRAPPER_H
#define DATA_WRAPPER_H

#include "misc/kfold.h"
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
    void shuffle(){
        auto& images = training_images;
        auto& labels = training_labels;
        PermutationMatrix<Dynamic,Dynamic> perm(images.rows());
        perm.setIdentity();
        std::random_shuffle(perm.indices().data(), perm.indices().data()+perm.indices().size());
        images = perm * images; // permute rows
        labels = perm * labels;
    }
protected:
    void readCSV(const string& path, vector<int>& label, vector< vector<float> >& data){
        ifstream file (path);
        if (file.is_open()) {
            string str;
            while(getline(file,str)){
                istringstream ss(str);
                data.push_back(vector<float>());
                label.push_back(0);
                readLine(ss,label.back(),data.back());
            }
            file.close();
        } else {
            cout << "Couldnt read csv " << path << endl;
            exit(1);
        }
    }  
    
    Eigen::MatrixXf ConvertToEigenMatrix(std::vector<std::vector<float>>& data)
    {
        Eigen::MatrixXf eMatrix(data.size(), data[0].size());
        for (int i = 0; i < data.size(); ++i)
            eMatrix.row(i) = Eigen::VectorXf::Map(&data[i][0], data[0].size());
        return eMatrix;
    }
    
    Eigen::VectorXi ConvertToEigenVec(vector<int>& data){
        Eigen::VectorXi mat = VectorXi::Map(&data[0],data.size());
        return mat;
    }
private:
    void readLine(istringstream& ss, int& label, vector<float>& data){
       string str;
       getline(ss,str,',');
       label = atoi(str.c_str());
       while(getline(ss,str,',')){
           data.push_back(stof(str.c_str()));
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
        
        training_images = ConvertToEigenMatrix(tmp_training_images);
        training_labels = ConvertToEigenVec(tmp_training_labels);
        
        testing_images = ConvertToEigenMatrix(tmp_testing_images);
        testing_labels = ConvertToEigenVec(tmp_testing_labels);
    }
};

struct MNISTDataWrapper: public DataWrapper{   
    MNISTDataWrapper(const string& training_path, const string& testing_path){
        // test set
        vector< vector<float> > tmp_testing_images;
        vector<int> tmp_testing_labels;
        readCSV(testing_path,tmp_testing_labels,tmp_testing_images); 
        
        testing_images = ConvertToEigenMatrix(tmp_testing_images);
        testing_labels = ConvertToEigenVec(tmp_testing_labels);
        
        // training set
        vector< vector<float> > tmp_training_images;
        vector<int> tmp_training_labels;
        vector< vector<float> > tmp_validation_images;
        vector<int> tmp_validation_labels;
        
        vector< vector<float> > tmp_images;
        vector<int> tmp_labels;
        readCSV(training_path,tmp_labels,tmp_images);
        
        // kfold
        std::vector<int> indicies(tmp_labels.size());
        std::iota(indicies.begin(), indicies.end(), 0);
        const int folds = 10;
        Kfold<vector<int>::const_iterator> kf(folds, indicies.begin(), indicies.end());

        vector<int> training, validation; // indicies
        kf.getFold(folds, back_inserter(training), back_inserter(validation));

        for(auto x: training){
            tmp_training_labels.push_back(tmp_labels[x]);
            tmp_training_images.push_back(tmp_images[x]);
        }
        training_images = ConvertToEigenMatrix(tmp_training_images);
        training_labels = ConvertToEigenVec(tmp_training_labels);
        training_images = training_images/255;
        
        for(auto x: validation){
            tmp_validation_labels.push_back(tmp_labels[x]);
            tmp_validation_images.push_back(tmp_images[x]);
        }
        validation_images = ConvertToEigenMatrix(tmp_validation_images);
        validation_labels = ConvertToEigenVec(tmp_validation_labels);
        validation_images = validation_images/255;
    }
};


#endif /* DATA_WRAPPER_H */

