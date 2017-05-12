
#include "nodes/bias_node.h"
#include "messages/forward_propagation_message.h"
#include "messages/backward_propagation_message.h"

std::string BiasNode::m_type = "Bias";
NodeRegister<BiasNode> BiasNode::m_reg(BiasNode::m_type);

bool BiasNode::sendForwardMsgs(vector<Message*>& msgs){
    assert(readyToSendForward());
    
    if(weights.empty()) initWeights();
    
    msgs.reserve(outgoingForwardEdges.size());
    assert(weights.size() == 1);
    for(unsigned i = 0; i < outgoingForwardEdges.size(); i++){
        assert( 0 == outgoingForwardEdges[i]->msgStatus );
        auto msg = forwardMessagePool->getMessage();
        msg->src = m_id;
        msg->dst = outgoingForwardEdges[i]->dst->getId();
        msg->activation = output*weights[0];
        msgs.push_back(msg);
    }
    
    forwardSeenCount = 0;
}

bool BiasNode::sendBackwardMsgs(vector<Message*>& msgs){
    assert(readyToSendBackward());
    
    // perform weights update first
    settings->trainingStrategy->computeDeltaWeights(settings,output,deltas,deltaWeights);
    
    newWeights[0] += deltaWeights[0]; // update step  
    
    msgs.reserve(outgoingBackwardEdges.size());
    for(unsigned i = 0; i < outgoingBackwardEdges.size(); i++){
        assert( 0 == outgoingBackwardEdges[i]->msgStatus );        
        auto msg = backwardMessagePool->getMessage();
        msg->src = m_id;
        msg->dst = outgoingBackwardEdges[i]->dst->getId();
        msgs.push_back(msg);
    }
    assert(msgs.size() == 1);
    
    backwardSeenCount = 0;
}


void BiasNode::onRecv(ForwardPropagationMessage* msg) {
    // notifying msg from sync node
    forwardSeenCount++;

    forwardMessagePool->returnMessage(msg);
    
    // weights update step
    if(readyToSendForward())
        weights = newWeights;
}

void BiasNode::onRecv(BackwardPropagationMessage* msg) {
    deltas[0] += msg->delta;
    backwardSeenCount++;
    
    backwardMessagePool->returnMessage(msg);
}
