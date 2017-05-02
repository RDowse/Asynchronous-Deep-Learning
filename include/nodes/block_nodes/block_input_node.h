
/* 
 * File:   block_input_node.h
 * Author: ryan
 *
 * Created on 01 May 2017, 17:35
 */

#ifndef BLOCK_INPUT_NODE_H
#define BLOCK_INPUT_NODE_H

#include "nodes/block_nodes/block_neural_node.h"
#include "graphs/graph_settings.h"
#include "misc/node_factory.h"
#include "tools/logging.h"
#include "tools/math.h"

#include <Eigen/Dense>

class BlockNeuralNode::InputNode : public BlockNeuralNode{
    static NodeRegister<BlockNeuralNode::InputNode> m_reg;
    static std::string m_type;
    
    map<int,int> dstWeightIndex;        // map of weights associated to dst ids
    
    Eigen::MatrixXf deltas;
    Eigen::MatrixXf deltaWeights;     // delta weights, for momentum
    Eigen::MatrixXf newWeights;       // intermediate updated weights
    Eigen::MatrixXf weights;
public:
    InputNode(shared_ptr<GraphSettings> graphSettings): BlockNeuralNode(graphSettings){}
    virtual ~InputNode(){}
    string getType() override {return InputNode::m_type;}

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

#endif /* BLOCK_INPUT_NODE_H */

