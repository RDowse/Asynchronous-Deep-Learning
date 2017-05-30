/* 
 * File:   input_node.h
 * Author: ryan
 *
 * Created on 23 March 2017, 23:36
 */

#ifndef PARALLEL_DATA_INPUT_NODE_H
#define PARALLEL_DATA_INPUT_NODE_H

#include "nodes/pardata_nodes/parallel_data_neural_node.h"
#include "misc/node_factory.h"
#include "tools/logging.h"
#include "tools/math.h"
#include "common.h"

#include <stack>
#include <string>
#include <cassert>
#include <cstdio>
#include <exception>
#include <memory>
#include <typeinfo>
#include <random>
#include <functional>

using namespace std;

class ParallelDataNeuralNode::InputNode: public ParallelDataNeuralNode{
    static NodeRegister<InputNode> m_reg;
    static std::string m_type;
    
    unordered_map<int,int> dstWeightIndex;        // map of weights associated to dst ids
    
    Eigen::MatrixXf receivedDelta;
    Eigen::VectorXf deltaWeights;     // delta weights, for momentum
    Eigen::VectorXf newWeights;       // intermediate updated weights
    Eigen::VectorXf weights;
public:
    InputNode(shared_ptr<GraphSettings> context): ParallelDataNeuralNode(context){}
    virtual ~InputNode(){}
    string getType() override {return InputNode::m_type;}

    void addEdge(Edge* e) override;
    
    void setWeights(const vector<float>& w) override{
        assert(w.size() == outgoingForwardEdges.size());
        //weights = Eigen::Map<Eigen::VectorXf>(&w[0],w.size());
        newWeights = weights; 
        
        // init size of delta values
        receivedDelta = Eigen::VectorXf::Zero(weights.size());
        deltaWeights = Eigen::VectorXf::Zero(weights.size());
    }
        
    void onRecv(ForwardPropagationMessage* msg) override;
    void onRecv(BackwardPropagationMessage* msg) override;

    bool sendForwardMsgs(vector<Message*>& msgs) override;
    bool sendBackwardMsgs(vector<Message*>& msgs) override;    
private:
    // for populating weights map
    int map_index = 0;
    void initWeights(){
        weights = Eigen::VectorXf::Zero(outgoingForwardEdges.size());
        context->initWeightsFnc(weights,outgoingForwardEdges.size(),incomingForwardEdges.size());
        newWeights = weights;    
        
        // init size of delta values
        deltaWeights = Eigen::VectorXf::Zero(weights.size());
    }
};

#endif /* INPUT_NODE_H */

