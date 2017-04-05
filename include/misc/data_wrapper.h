/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   data_wrapper.h
 * Author: ryan
 *
 * Created on 02 April 2017, 14:39
 */

#ifndef DATA_WRAPPER_H
#define DATA_WRAPPER_H

#include "mnist/mnist_reader.hpp"

#include <vector>

using namespace std;

template<typename TDataSet, typename TDataSub, typename TLabel>
struct DataWrapper{
    TDataSet m_dataSet;
    DataWrapper(TDataSet dataSet):m_dataSet(dataSet){}
    
    virtual const vector<TDataSub>& getData()=0;
    virtual const vector<TLabel>& getLabels()=0;
};

typedef mnist::MNIST_dataset<vector, vector<uint8_t>, uint8_t> MNISTDataset;

struct MNISTDatasetWrapper : DataWrapper<MNISTDataset, vector<uint8_t>, uint8_t>{
    
    MNISTDatasetWrapper(MNISTDataset dataSet):DataWrapper(dataSet){}
    
    const vector<vector<uint8_t>>& getData(){
        return m_dataSet.training_images;
    }
    
    const vector<uint8_t>& getLabels(){
        return m_dataSet.training_labels;
    }
};

#endif /* DATA_WRAPPER_H */

