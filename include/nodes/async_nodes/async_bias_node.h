/* 
 * File:   bias_node.h
 * Author: ryan
 *
 * Created on 04 April 2017, 22:54
 */

#ifndef ASYNC_BIAS_NODE_H
#define ASYNC_BIAS_NODE_H

#include "nodes/async_nodes/async_neural_node.h"
#include "misc/node_factory.h"
#include "graphs/graph_settings.h"
#include "graphs/dnn_graph_settings.h"
#include "states/backward_train_state.h"
#include "states/forward_train_state.h"
#include "tools/math.h"

#include <Eigen/Dense>
#include <stack>
#include <cassert>

using namespace std;

class AsyncNeuralNode::BiasNode : public AsyncNeuralNode{
    // for populating weights map
    int map_index = 0;
    unordered_map<int,int> dstWeightIndex;        // map of weights associated to dst ids
    
    Eigen::MatrixXf receivedDelta;
    Eigen::VectorXf newWeights;
    Eigen::VectorXf deltaWeights;
    Eigen::VectorXf weights; 
    
    Eigen::VectorXf input; // set to 1
public:
    static std::string m_type;
    BiasNode(shared_ptr<GraphSettings> context): AsyncNeuralNode(context){};
    virtual ~BiasNode(){}
    string getType() override {return BiasNode::m_type;}
    void addEdge(Edge* e) override;
    void setWeights(const vector<float>& w) override{  }
    
    void onRecv(ForwardPropagationMessage* msg) override;
    void onRecv(BackwardPropagationMessage* msg) override;
    
    bool sendBackwardMsgs(vector<Message*>& msgs) override;
    bool sendForwardMsgs(vector<Message*>& msgs) override;
private:
    void initWeights(){
        weights = Eigen::VectorXf::Zero(outgoingForwardEdges.size());
        context->initWeightsFnc(weights,outgoingForwardEdges.size(),incomingForwardEdges.size());
        newWeights = weights;    
        
        // init size of delta values
        deltaWeights = Eigen::VectorXf::Zero(weights.size());
    }
};


#endif /* BIAS_NODE_H */

