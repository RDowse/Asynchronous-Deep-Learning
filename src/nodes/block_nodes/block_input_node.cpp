
#include "nodes/block_nodes/block_input_node.h"
#include "messages/forward_propagation_message.h"
#include "messages/backward_propagation_message.h"
#include "tools/math.h"

std::string BlockNeuralNode::InputNode::m_type = "BlockInputNode";
NodeRegister<BlockNeuralNode::InputNode> BlockNeuralNode::InputNode::m_reg(BlockNeuralNode::InputNode::m_type);

void BlockNeuralNode::InputNode::addEdge(Edge* e) {
     // add to original edge sets
     Node::addEdge(e);
 }

bool BlockNeuralNode::InputNode::sendForwardMsgs(vector<Message*>& msgs){
    assert(readyToSendForward());

    forwardSeenCount = 0;
}

bool BlockNeuralNode::InputNode::sendBackwardMsgs(vector<Message*>& msgs){
    assert(readyToSendBackward());
    
    // reset
    backwardSeenCount = 0;
}

void BlockNeuralNode::InputNode::onRecv(ForwardPropagationMessage* msg){
    output = msg->matActivation;
    forwardSeenCount++;
    
    delete msg;
    
    // weight update step
    if(readyToSendForward() && settings->update)
        weights = newWeights;
} 

void BlockNeuralNode::InputNode::onRecv(BackwardPropagationMessage* msg) {
    int index = dstWeightIndex[msg->src];
//    deltas[index] = msg->delta;
    backwardSeenCount++;
    
    delete msg;
}