
#include "nodes/input_node.h"
#include "messages/forward_propagation_message.h"
#include "messages/backward_propagation_message.h"
#include "tools/math.h"

std::string NeuralNode::InputNode::m_type = "Input";
NodeRegister<NeuralNode::InputNode> NeuralNode::InputNode::m_reg(NeuralNode::InputNode::m_type);

void NeuralNode::InputNode::addEdge(Edge* e) {
     // add to original edge sets
     Node::addEdge(e);
     // check edge belongs to this node
     if(e->src->getId() == m_id){
         if(e->dst->getType() == "Sync"){
             outgoingBackwardEdges.push_back(e);
         } else if(e->dst->getType() == "Hidden"
                 || e->dst->getType() == "Output"){
             outgoingForwardEdges.push_back(e);
             dstWeightIndex[e->dst->getId()] = map_index++;
         } else {
             cout << "Unknown type " << e->dst->getType() << "\n";
             assert(0);
         }
     } else if(e->dst->getId() == m_id){
         if(e->src->getType() == "Sync"){
             incomingForwardEdges.push_back(e);
         } else if(e->src->getType() == "Hidden"
                 || e->src->getType() == "Output"){
             incomingBackwardEdges.push_back(e);
         } else {
             cout << "Unknown type " << e->dst->getType() << "\n";
             assert(0);
         }
     } 
     
     // tmp tests
     assert(outgoingBackwardEdges.size() <= 1);        
     assert(incomingForwardEdges.size() <= 1);
 }

bool NeuralNode::InputNode::sendForwardMsgs(vector<Message*>& msgs){
    assert(readyToSendForward());
    
    if(weights.empty()) initWeights();
    
    Logging::log(3, "%s node %d input: %f", m_type.c_str(), m_id, output);
    
    //msgs.reserve(weights.size());
    assert(weights.size() == outgoingForwardEdges.size());
    for(unsigned i = 0; i < outgoingForwardEdges.size(); i++){
        assert( 0 == outgoingForwardEdges[i]->msgStatus );
        auto msg = new ForwardPropagationMessage();
        msg->src = m_id;
        msg->dst = outgoingForwardEdges[i]->dst->getId();
        msg->activation = output*weights[i];
        msgs.push_back(msg);
    }
    
    // reset 
    forwardSeenCount = 0;
}

bool NeuralNode::InputNode::sendBackwardMsgs(vector<Message*>& msgs){
    assert(readyToSendBackward());
            
    // perform weight update first
    settings->trainingStrategy->computeDeltaWeights(settings,output,deltas,deltaWeights);
    
    for(int i = 0; i < deltaWeights.size(); ++i)
        newWeights[i] += deltaWeights[i]; // update step    
    
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

void NeuralNode::InputNode::onRecv(ForwardPropagationMessage* msg){
    output = msg->activation;
    forwardSeenCount++;
    
    delete msg;
    
    // weight update step
    if(readyToSendForward())
        weights = newWeights;
} 

void NeuralNode::InputNode::onRecv(BackwardPropagationMessage* msg) {
    int index = dstWeightIndex[msg->src];
    deltas[index] = msg->delta;
    backwardSeenCount++;
    
    delete msg;
}