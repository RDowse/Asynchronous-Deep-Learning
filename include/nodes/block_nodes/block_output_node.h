
/* 
 * File:   block_output_node.h
 * Author: ryan
 *
 * Created on 01 May 2017, 18:30
 */

#ifndef BLOCK_OUTPUT_NODE_H
#define BLOCK_OUTPUT_NODE_H

#include "nodes/block_nodes/block_neural_node.h"
#include "graphs/graph_settings.h"
#include "misc/node_factory.h"
#include "tools/logging.h"
#include "tools/math.h"

#include <Eigen/Dense>

class BlockNeuralNode::OutputNode: public BlockNeuralNode{
    static NodeRegister<BlockNeuralNode::OutputNode> m_reg;
    static std::string m_type;
    
    Eigen::MatrixXf error;
    Eigen::MatrixXf target;
    Eigen::MatrixXf value;
public:
    OutputNode(shared_ptr<GraphSettings> context): BlockNeuralNode(context){
        layer = 2; // TODO correct so multiple layers can be added
    }
    virtual ~OutputNode(){}
    string getType() override{return OutputNode::m_type;}
    
    void addEdge(Edge* e) override;

    void onRecv(ForwardPropagationMessage* msg) override;
    void onRecv(BackwardPropagationMessage* msg) override;
    
    bool sendForwardMsgs(vector<Message*>& msgs) override;
    bool sendBackwardMsgs(vector<Message*>& msgs) override;
private:
    void initOutput(){
        int blockSize = settings->blockTopology[layer].front();
        int batchSize = settings->miniBatchSize;
        output = MatrixXf(blockSize,batchSize);
    }
    void initTarget(){
        int blockSize = settings->blockTopology[layer].front();
        int batchSize = settings->miniBatchSize;
        target = MatrixXf(blockSize,batchSize);
    }
};


#endif /* BLOCK_OUTPUT_NODE_H */

