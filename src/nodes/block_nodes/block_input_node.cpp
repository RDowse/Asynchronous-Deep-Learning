
#include "nodes/block_nodes/block_input_node.h"
#include "messages/forward_propagation_message.h"
#include "messages/backward_propagation_message.h"
#include "tools/math.h"

std::string BlockNeuralNode::InputNode::m_type = "BlockInput";
NodeRegister<BlockNeuralNode::InputNode> BlockNeuralNode::InputNode::m_reg(BlockNeuralNode::InputNode::m_type);

void BlockNeuralNode::InputNode::addEdge(Edge* e) {
    // add to original edge sets
    Node::addEdge(e);
    if(e->src->getId() == m_id){
         if(e->dst->getType() == "BlockSync"){
             outgoingBackwardEdges.push_back(e);
         } else if(e->dst->getType() == "BlockHidden"
                 || e->dst->getType() == "BlockOutput"){
             outgoingForwardEdges.push_back(e);
             dstWeightIndex[e->dst->getId()] = map_index++;
         } else {
             cout << "Unknown type " << e->dst->getType() << "\n";
             assert(0);
         }
     } else if(e->dst->getId() == m_id){
         if(e->src->getType() == "BlockSync"){
             incomingForwardEdges.push_back(e);
         } else if(e->src->getType() == "BlockHidden"
                 || e->src->getType() == "BlockOutput"){
             incomingBackwardEdges.push_back(e);
         } else {
             cout << "Unknown type " << e->src->getType() << "\n";
             assert(0);
         }
     } 
     
     // tmp tests
     assert(outgoingBackwardEdges.size() <= 1);        
     assert(incomingForwardEdges.size() <= 1);
 }

bool BlockNeuralNode::InputNode::sendForwardMsgs(vector<Message*>& msgs){
    assert(readyToSendForward());
    
    // checks
    assert(output.size() != 0);
    
    if(!weights.size()) initWeights();
    
    //Logging::log(3, "%s node %d input: %f", m_type.c_str(), m_id, output);
    
    // sigmoid calculations
    for(int col = 0; col < output.cols(); ++col)
        for(int row = 0; row < output.rows(); ++row)
            output(row,col) = settings->activationFnc(output(row,col));
    
    int blockIndex = 0;        
    
    //msgs.reserve(weights.size());
    //assert(weights.size() == outgoingForwardEdges.size());
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
    
    // reset output, TODO: refactor if inefficient
    output.Zero(output.rows(),output.cols());
    
    // reset 
    forwardSeenCount = 0;
}

bool BlockNeuralNode::InputNode::sendBackwardMsgs(vector<Message*>& msgs){
    assert(readyToSendBackward());
            
    // perform weight update first
    //settings->trainingStrategy->computeDeltaWeights(settings,output,deltas,deltaWeights);
    
//    for(int i = 0; i < deltaWeights.size(); ++i)
//        newWeights[i] += deltaWeights[i]; // update step    
    
    for(unsigned i = 0; i < outgoingBackwardEdges.size(); i++){
        assert( 0 == outgoingBackwardEdges[i]->msgStatus );
        auto msg = new BackwardPropagationMessage();
        msg->src = m_id;
        msg->dst = outgoingBackwardEdges[i]->dst->getId();
        msgs.push_back(msg);
    }
    assert(msgs.size() == 1);
    
    // reset
    backwardSeenCount = 0;
}

void BlockNeuralNode::InputNode::onRecv(ForwardPropagationMessage* msg){
    if(!output.size()) initOutput();
    output += msg->matActivation;
    forwardSeenCount++;
    
    delete msg;
    
    // weight update step
    if(readyToSendForward() && settings->update)
        weights = newWeights;
} 

void BlockNeuralNode::InputNode::onRecv(BackwardPropagationMessage* msg) {
    int index = dstWeightIndex[msg->src];
    //deltas[index] = msg->delta;
    backwardSeenCount++;
    
    delete msg;
}