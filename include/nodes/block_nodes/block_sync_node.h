
/* 
 * File:   block_sync_node.h
 * Author: ryan
 *
 * Created on 03 May 2017, 23:36
 */

#ifndef BLOCK_SYNC_NODE_H
#define BLOCK_SYNC_NODE_H

#include "nodes/block_nodes/block_neural_node.h"
#include "graphs/graph_settings.h"
#include "misc/node_factory.h"
#include "graphs/dnn_graph_settings.h"

#include "misc/data_wrapper.h"

#include <Eigen/Dense>

#include <string>
#include <cassert>
#include <cstdio>
#include <exception>
#include <memory>
#include <typeinfo>
#include <limits>

using namespace std;

class BlockNeuralNode::SyncNode: public BlockNeuralNode{
    static NodeRegister<SyncNode> m_reg;
    
    DataWrapper* dataset;
    
    int nInput;                 // number of input nodes
    int nOutput;                // number of output nodes
    
    bool tick = true;           // trigger initial message propagation
    bool validating = false;    // flag for propagating validation set
  
    int sampleIndex = 0;
    int epochCount = 0;
    
    vector<float> min_error; 
    vector<float> error;
    
    vector<int> trainingIndices;
    
    map<int,int> dstOutputIndex;        // map backprop index to output
    
    float actMax = settings->activationFnc(10);
    float actMin = settings->activationFnc(-10);
    
    float training_error = 0;
public:
    static std::string m_type;
    SyncNode(shared_ptr<GraphSettings> settings): BlockNeuralNode(settings){}
    virtual ~SyncNode(){}
    
    string getType() override {return SyncNode::m_type;}
    void setDataSet(DataWrapper* ds ){
        dataset = ds;
        for(int i = 0; i < dataset->training_images.cols(); ++i){
            trainingIndices.push_back(i);
        }
            
        min_error = vector<float>(dataset->training_images.cols(),std::numeric_limits<float>::max());
        error = vector<float>(dataset->training_images.cols(),std::numeric_limits<float>::max());
    }
    void addEdge(Edge* e) override;    
    
    bool readyToSendForward() override;
    bool readyToSendBackward() override;
    
    void onRecv(ForwardPropagationMessage* msg) override;
    void onRecv(BackwardPropagationMessage* msg) override;
    
    bool sendBackwardMsgs(vector<Message*>& msgs);
    bool sendForwardMsgs(vector<Message*>& msgs);
    
private:
    int map_index = 0;
    void initOutput(){
        int blockSize = settings->netTopology.back();
        int batchSize = settings->miniBatchSize;
        output = MatrixXf(blockSize,batchSize);
    }
};


#endif /* BLOCK_SYNC_NODE_H */

