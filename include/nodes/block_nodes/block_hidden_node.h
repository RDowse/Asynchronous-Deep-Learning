
/* 
 * File:   block_hidden_node.h
 * Author: ryan
 *
 * Created on 01 May 2017, 18:30
 */

#ifndef BLOCK_HIDDEN_NODE_H
#define BLOCK_HIDDEN_NODE_H

#include "nodes/block_nodes/block_neural_node.h"
#include "graphs/graph_settings.h"
#include "misc/node_factory.h"
#include "tools/logging.h"
#include "tools/math.h"

#include <Eigen/Dense>

class BlockNode::HiddenNode: public BlockNeuralNode{
    static NodeRegister<BlockNode::HiddenNode> m_reg;
    static std::string m_type;
    
    map<int,int> dstWeightIndex;        // map backprop index to the relevant weight
    
    Eigen::MatrixXf deltas;           // store received delta values
    Eigen::MatrixXf deltaWeights;     // delta weights, for momentum
    Eigen::MatrixXf newWeights;       // new weights to update
    Eigen::MatrixXf weights;
   
    float value = 0;
    float error = 0;
public:
    HiddenNode(shared_ptr<GraphSettings> context): BlockNode::NeuralNode(context){}
    virtual ~HiddenNode(){}
    string getType() override {return BlockNode::HiddenNode::m_type;}
    
    void addEdge(Edge* e) override;

    void onRecv(ForwardPropagationMessage* msg) override;
    void onRecv(BackwardPropagationMessage* msg) override;

    bool sendForwardMsgs(vector<Message*>& msgs) override;
    bool sendBackwardMsgs(vector<Message*>& msgs) override;
private:
    // for populating weights map
    int map_index = 0;
    void initWeights(){}
};


#endif /* BLOCK_HIDDEN_NODE_H */

