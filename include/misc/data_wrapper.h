/* 
 * File:   data_wrapper.h
 * Author: ryan
 *
 * Created on 02 April 2017, 14:39
 */

#ifndef DATA_WRAPPER_H
#define DATA_WRAPPER_H

#include "tools/math.h"

#include <vector>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

using namespace std;

struct DataWrapper{
    
    vector<int> training_labels;
    vector< vector<int> > training_images;
    
    vector<int> validation_labels;
    vector< vector<int> > validation_images;
    
    vector<int> testing_labels;
    vector< vector<int> > testing_images;
    
    DataWrapper(const string& training_path, const string& testing_path){
        vector< vector<int> > tmp_images;
        vector<int> tmp_labels;
        //readCSV(training_path,training_labels,training_images);
        readCSV(training_path,tmp_labels,tmp_images);
        readCSV(testing_path,testing_labels,testing_images);
        
        training_labels.reserve(tmp_labels.size());
        training_images.reserve(tmp_labels.size());
        
        std::vector<int> indicies(tmp_labels.size());
        std::iota(indicies.begin(), indicies.end(), 0);

        const int folds = 5;
        math::Kfold<vector<int>::const_iterator> kf(folds, indicies.begin(), indicies.end());

        vector<int> train, test;
        
        int i = 0;
        kf.getFold(i + 1, back_inserter(train), back_inserter(test));

        cout << "Fold " << i + 1 << " Training Data" << endl;
        for(auto x: train){
            training_labels.push_back(tmp_labels[x]);
            training_images.push_back(tmp_images[x]);
            cout << tmp_labels[x] << " ";
        }
        cout << endl;
        cout << "Fold " << i + 1 << " Testing Data" << endl;
        for(auto x: test){
            validation_labels.push_back(tmp_labels[x]);
            validation_images.push_back(tmp_images[x]);
            cout << tmp_labels[x] << " ";
        }
        cout << endl;
    }
private:
    
    float ratio = 0.8; // ratio training to crossvalidation
    
    void readLine(istringstream& ss, int& label, vector<int>& data){
       string str;
       getline(ss,str,',');
       label = atoi(str.c_str());
       while(getline(ss,str,',')){
           data.push_back(atoi(str.c_str()));
       }
    }

   void readCSV(const string& path, vector<int>& label, vector< vector<int> >& data){
       ifstream file (path);
       string str;
       while(getline(file,str)){
           istringstream ss(str);
           data.push_back(vector<int>());
           label.push_back(0);
           readLine(ss,label.back(),data.back());
       }
    }
};


#endif /* DATA_WRAPPER_H */

