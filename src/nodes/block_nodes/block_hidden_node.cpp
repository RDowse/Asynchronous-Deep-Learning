

#include "nodes/block_nodes/block_hidden_node.h"
#include "messages/forward_propagation_message.h"
#include "messages/backward_propagation_message.h"

std::string BlockNode::HiddenNode::m_type = "BlockHidden";
NodeRegister<BlockNode::HiddenNode> BlockNode::HiddenNode::m_reg(BlockNode::HiddenNode::m_type);

void BlockNode::HiddenNode::addEdge(Edge* e) {
    // add to original edge sets
    Node::addEdge(e);
//    if(e->src->getId() == m_id){
//        if(e->dst->getId() > m_id){ // change based on type of edge.
//            outgoingForwardEdges.push_back(e);
//            dstWeightIndex[e->dst->getId()] = map_index++;
//        } else {
//            outgoingBackwardEdges.push_back(e);
//        }
//    } else if(e->dst->getId() == m_id){
//        if(e->src->getId() < m_id){ // change based on type of edge.
//            incomingForwardEdges.push_back(e);
//        } else {
//            incomingBackwardEdges.push_back(e);
//        }
//    }
}

bool BlockNode::HiddenNode::sendForwardMsgs(vector<Message*>& msgs) {
    assert(readyToSendForward());
    
//    if(weights.empty()) initWeights();
//    
//    // calulate output activation
//    output = settings->activationFnc(value);
//    
//    //msgs.reserve(weights.size());
//    assert(weights.size() == outgoingForwardEdges.size());
//    for(unsigned i = 0; i < outgoingForwardEdges.size(); i++){
//        assert( 0 == outgoingForwardEdges[i]->msgStatus );
//        auto msg = new ForwardPropagationMessage();
//        msg->src = m_id;
//        msg->dst = outgoingForwardEdges[i]->dst->getId();
//        msg->activation = output*weights[i];
//        msgs.push_back(msg);
//    }
//    
//    //send(msgs,outgoingForwardEdges);
    
    // reset
    //value = 0;
    forwardSeenCount = 0;
}

bool BlockNode::HiddenNode::sendBackwardMsgs(vector<Message*>& msgs){
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

void BlockNode::HiddenNode::onRecv(ForwardPropagationMessage* msg) {
//    value += msg->matActivation;
    forwardSeenCount++;
    
    delete msg;
    
    // weight update step
//    if(readyToSendForward() && settings->update)
//        weights = newWeights;
}

void BlockNode::HiddenNode::onRecv(BackwardPropagationMessage* msg) {
//    int index = dstWeightIndex[msg->src];
//    deltas[index] = msg->delta;
//    backwardSeenCount++;
    
    delete msg;
}