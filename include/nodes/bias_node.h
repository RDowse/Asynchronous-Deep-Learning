/* 
 * File:   bias_node.h
 * Author: ryan
 *
 * Created on 04 April 2017, 22:54
 */

#ifndef BIAS_NODE_H
#define BIAS_NODE_H

#include "nodes/neural_node.h"
#include "misc/node_factory.h"
#include "graphs/graph_settings.h"
#include "graphs/dnn_graph_settings.h"
#include "tools/math.h"

#include <stack>
#include <cassert>

using namespace std;

class BiasNode : public NeuralNode{
    static NodeRegister<BiasNode> m_reg;
    static std::string m_type;
    
    // for populating weights map
    int map_index = 0;
    unordered_map<int,int> dstWeightIndex;        // map of weights associated to dst ids
    
    Eigen::MatrixXf receivedDelta;
    Eigen::VectorXf newWeights;
    Eigen::VectorXf deltaWeights;
    Eigen::VectorXf weights; 
    
    Eigen::VectorXf input; // set to 1
public:
    BiasNode(shared_ptr<GraphSettings> context): NeuralNode(context){};
    virtual ~BiasNode(){}
    string getType() override {return BiasNode::m_type;}
    void addEdge(Edge* e) override;
    void setWeights(const vector<float>& w) override{
        assert(w.size() == 1);
        //weights = Eigen::Map<Eigen::VectorXf>(&w[0],w.size());
        newWeights = weights; 
        
        activation = Eigen::VectorXf::Ones(weights.size());
        
        // init size of delta values
        receivedDelta = Eigen::VectorXf(weights.size());
        deltaWeights = Eigen::VectorXf(weights.size());
    }
    
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

