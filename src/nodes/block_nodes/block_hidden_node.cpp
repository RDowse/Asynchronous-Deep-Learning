

#include "nodes/block_nodes/block_hidden_node.h"
#include "messages/forward_propagation_message.h"
#include "messages/backward_propagation_message.h"

std::string BlockNeuralNode::HiddenNode::m_type = "BlockHidden";
NodeRegister<BlockNeuralNode::HiddenNode> BlockNeuralNode::HiddenNode::m_reg(BlockNeuralNode::HiddenNode::m_type);

void BlockNeuralNode::HiddenNode::addEdge(Edge* e) {
    // add to original edge sets
    Node::addEdge(e);
    if(e->src->getId() == m_id){
        if(e->dst->getId() > m_id){ // change based on type of edge.
            outgoingForwardEdges.push_back(e);
            dstWeightIndex[e->dst->getId()] = map_index++;
        } else {
            outgoingBackwardEdges.push_back(e);
        }
    } else if(e->dst->getId() == m_id){
        if(e->src->getId() < m_id){ // change based on type of edge.
            incomingForwardEdges.push_back(e);
        } else {
            incomingBackwardEdges.push_back(e);
        }
    }
}

bool BlockNeuralNode::HiddenNode::sendForwardMsgs(vector<Message*>& msgs) {
    assert(readyToSendForward());
    
    if(!weights.size()) initWeights();
    
    // sigmoid calculations
    for(int col = 0; col < output.cols(); ++col)
        for(int row = 0; row < output.rows(); ++row)
            output(row,col) = settings->activationFnc(output(row,col));

    // index for current block of weights to use
    int blockIndex = 0;    
    
    for(unsigned i = 0; i < outgoingForwardEdges.size(); i++){
        assert( 0 == outgoingForwardEdges[i]->msgStatus );
        auto msg = new ForwardPropagationMessage();
        msg->src = m_id;
        msg->dst = outgoingForwardEdges[i]->dst->getId();
        
        // get block size of the next layer
        int blockSize = settings->blockTopology[layer+1][i];
        // separate output into blocks
        msg->matActivation = weights.block(0,blockIndex,weights.rows(),blockSize).transpose()*output;
        blockIndex+=blockSize;
        msgs.push_back(msg);
    }
    
    // reset
    output.Zero(output.rows(),output.cols());
    forwardSeenCount = 0;
}

bool BlockNeuralNode::HiddenNode::sendBackwardMsgs(vector<Message*>& msgs){
    assert(readyToSendBackward());

    // perform weight update
//    float delta_sum = 0;
//    for(int i = 0; i < weights.size(); ++i)
//        delta_sum += deltas[i]*weights[i];
//    
//    // perform weight update first
//    settings->trainingStrategy->computeDeltaWeights(settings,output,deltas,deltaWeights);
//        
//    for(int i = 0; i < deltaWeights.size(); ++i)
//        newWeights[i] += deltaWeights[i]; // update step  
//    
//    // dispatch msgs, calculating delta for next nodes
//    auto delta = delta_sum*settings->deltaActivationFnc(output);
//    for(unsigned i = 0; i < outgoingBackwardEdges.size(); i++){
//        assert( 0 == outgoingBackwardEdges[i]->msgStatus );
//        auto msg = new BackwardPropagationMessage();
//        msg->src = m_id;
//        msg->dst = outgoingBackwardEdges[i]->dst->getId();
//        msg->delta = delta; 
//        msgs.push_back(msg);
//    }
    
    // reset
    backwardSeenCount = 0;
}

void BlockNeuralNode::HiddenNode::onRecv(ForwardPropagationMessage* msg) {
    if(!output.size()) initOutput();
    output += msg->matActivation;
    forwardSeenCount++;
    
    delete msg;
    
    // weight update step
//    if(readyToSendForward() && settings->update)
//        weights = newWeights;
}

void BlockNeuralNode::HiddenNode::onRecv(BackwardPropagationMessage* msg) {
//    int index = dstWeightIndex[msg->src];
//    deltas[index] = msg->delta;
//    backwardSeenCount++;
    
    delete msg;
}