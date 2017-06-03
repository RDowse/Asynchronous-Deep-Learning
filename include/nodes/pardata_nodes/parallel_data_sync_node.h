/* 
 * File:   sync_node.h
 * Author: ryan
 *
 * Created on 13 April 2017, 00:47
 */

#ifndef PARALLEL_DATA_SYNC_NODE_H
#define PARALLEL_DATA_SYNC_NODE_H

#include "nodes/pardata_nodes/parallel_data_neural_node.h"
#include "graphs/graph_settings.h"
#include "misc/node_factory.h"
#include "graphs/dnn_graph_settings.h"

#include "misc/data_wrapper.h"

#include <unordered_map>
#include <stack>
#include <string>
#include <cassert>
#include <cstdio>
#include <exception>
#include <memory>
#include <typeinfo>
#include <limits>

using namespace std;

class ParallelDataNeuralNode::SyncNode: public ParallelDataNeuralNode{
    DataWrapper* dataset;
    
    bool tick = true;
    bool validating = false;    // flag for propagating validation set
    
    // sampling
    int sampleIndex = 0;
    int batchSize = 0;
    
    // batch tracking
    int batchCount = 0;
    int activeBatchCount = 0;
    int timer = 0;
    
    int map_index = 0;
    unordered_map<int,int> dstOutputIndex;        // map backprop index to output
    
    // Error calculations
    std::vector<Eigen::MatrixXf,Eigen::aligned_allocator<Eigen::MatrixXf> > receivedOutput;
    float training_error = 0;
    float accuracy = 0;
    vector<float> min_error; 
    vector<float> error;
public:
    static std::string m_type;
    SyncNode(shared_ptr<GraphSettings> context): ParallelDataNeuralNode(context){
        try{
            auto tmp_context = std::static_pointer_cast<DNNGraphSettings>(context);
            receivedOutput = vector<Eigen::MatrixXf,Eigen::aligned_allocator<Eigen::MatrixXf> >(tmp_context->numModels);
        } catch (const std::bad_cast& e) {
            std::cout << e.what() << "\n";
        }
    }
    virtual ~SyncNode(){}
    string getType() override {return SyncNode::m_type;}
    void setDataSet(DataWrapper* ds ){
        dataset = ds;
        min_error = vector<float>(dataset->training_labels.size(),std::numeric_limits<float>::max());
        error = vector<float>(dataset->training_labels.size(),std::numeric_limits<float>::max());
    }
    void addEdge(Edge* e) override;
    
    bool readyToSendForward(int i) override;
    bool readyToSendBackward(int i) override;
    
    void onRecv(ForwardPropagationMessage* msg) override;
    void onRecv(BackwardPropagationMessage* msg) override;
    
    bool sendBackwardMsgs(vector<Message*>& msgs, int stateIndex);
    bool sendForwardMsgs(vector<Message*>& msgs, int stateIndex);
};

#endif /* SYNC_NODE_H */