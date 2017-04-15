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
    
    virtual vector<TDataSub>& getData(int sample) const =0;
    virtual TLabel getLabel(int sample) const =0;
};

typedef mnist::MNIST_dataset<vector, vector<uint8_t>, uint8_t> MNISTDataset;

//struct MNISTDatasetWrapper : DataWrapper<MNISTDataset, uint8_t, uint8_t>{
//    
//    MNISTDatasetWrapper(MNISTDataset dataSet):DataWrapper(dataSet){}
//    
//    vector<uint8_t>& getData(int sample) const {
//        return m_dataSet.training_images[sample];
//    }
//    
//    uint8_t getLabel(int sample) const {
//        return m_dataSet.training_labels[sample];
//    }
//};

#endif /* DATA_WRAPPER_H */

