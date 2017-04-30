
#include "nodes/bias_node.h"
#include "messages/forward_propagation_message.h"
#include "messages/backward_propagation_message.h"

std::string BiasNode::m_type = "Bias";
NodeRegister<BiasNode> BiasNode::m_reg(BiasNode::m_type);

bool BiasNode::sendForwardMsgs(vector<Message*>& msgs){
    assert(readyToSendForward());
    
    if(weights.empty()) initWeights();
    
    //msgs.reserve(weights.size());
    assert(weights.size() == 1);
    for(unsigned i = 0; i < outgoingForwardEdges.size(); i++){
        assert( 0 == outgoingForwardEdges[i]->msgStatus );
        auto msg = new ForwardPropagationMessage();
        msg->src = m_id;
        msg->dst = outgoingForwardEdges[i]->dst->getId();
        msg->value = output*weights[0];
        msgs.push_back(msg);
    }
    
    //send(msgs,outgoingForwardEdges);
    
    forwardSeenCount = 0;
}

bool BiasNode::sendBackwardMsgs(vector<Message*>& msgs){
    assert(readyToSendBackward());
    
    deltas[0] /= outgoingForwardEdges.size();
    
    // perform weights update first
    settings->trainingStrategy->computeDeltaWeights(settings,output,deltas,deltaWeights);

    newWeights[0] += deltaWeights[0]; // update step  
    
    for(unsigned i = 0; i < outgoingBackwardEdges.size(); i++){
        assert( 0 == outgoingBackwardEdges[i]->msgStatus );        
        auto msg = new BackwardPropagationMessage();
        msg->src = m_id;
        msg->dst = outgoingBackwardEdges[i]->dst->getId();
        msgs.push_back(msg);
    }
    assert(msgs.size() == 1);
    
    //send(msgs,outgoingBackwardEdges);
    
    backwardSeenCount = 0;
}


void BiasNode::onRecv(ForwardPropagationMessage* msg) {
    // notifying msg from sync node
    forwardSeenCount++;

    delete msg;
    
    // weights update step
    if(readyToSendForward() && settings->update)
        weights = newWeights;
}

void BiasNode::onRecv(BackwardPropagationMessage* msg) {
    deltas[0] += msg->delta;
    backwardSeenCount++;
    
    delete msg;
}
