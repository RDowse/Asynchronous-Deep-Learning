
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

class BlockNeuralNode::HiddenNode: public BlockNeuralNode{
    static NodeRegister<BlockNeuralNode::HiddenNode> m_reg;
    static std::string m_type;
    
    map<int,int> dstWeightIndex;        // map backprop index to the relevant weight
    
    Eigen::MatrixXf deltas;           // store received delta values
    Eigen::MatrixXf deltaWeights;     // delta weights, for momentum
    Eigen::MatrixXf newWeights;       // new weights to update
    Eigen::MatrixXf weights;

    float value = 0;
    float error = 0;
public:   
    int layer;
    int layerPos;
    HiddenNode(shared_ptr<GraphSettings> context): BlockNeuralNode(context){
        layer = 1;
    }
    virtual ~HiddenNode(){}
    string getType() override {return BlockNeuralNode::HiddenNode::m_type;}
    
    void addEdge(Edge* e) override;

    void onRecv(ForwardPropagationMessage* msg) override;
    void onRecv(BackwardPropagationMessage* msg) override;

    bool sendForwardMsgs(vector<Message*>& msgs) override;
    bool sendBackwardMsgs(vector<Message*>& msgs) override;
private:
    // for populating weights map
    int map_index = 0;
    void initWeights(){
        // TODO add bias node
        int blockSize = settings->blockTopology[layer].front();
        int nextLayerSize = settings->netTopology[layer+1];
        weights = MatrixXf::Zero(blockSize,nextLayerSize);
        settings->initWeightsFnc(weights,1,nextLayerSize);
        newWeights = weights;
    }
    void initOutput(){
        int blockSize = settings->blockTopology[layer].front();
        int batchSize = settings->miniBatchSize;
        output = MatrixXf(blockSize,batchSize);
    }
    void initDeltas(){
        int batchSize = settings->miniBatchSize;
        int nextLayerSize = settings->netTopology[layer+1];
        deltas = MatrixXf(nextLayerSize,batchSize);    
    }
};


#endif /* BLOCK_HIDDEN_NODE_H */

