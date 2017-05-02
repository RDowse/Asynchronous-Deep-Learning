
#include "nodes/output_node.h"

#include "messages/forward_propagation_message.h"
#include "messages/backward_propagation_message.h"
#include "tools/math.h"

std::string OutputNode::m_type = "Output";
NodeRegister<OutputNode> OutputNode::m_reg(OutputNode::m_type);

void OutputNode::addEdge(Edge* e) {
    // add to original edge sets
    Node::addEdge(e);
    // check edge belongs to this node
    if(e->src->getId() == m_id){
        if(e->dst->getType() == "Hidden" ||
            e->dst->getType() == "Input" ||
            e->dst->getType() == "Bias"){
            outgoingBackwardEdges.push_back(e);
        } else if(e->dst->getType() == "Sync"){
            outgoingForwardEdges.push_back(e);
        } else {
            cout << "Unknown type " << e->dst->getType() << "\n";
            assert(0);
        }
    } else if(e->dst->getId() == m_id){
        if(e->src->getType() == "Hidden" ||
            e->src->getType() == "Input" ||
            e->src->getType() == "Bias"){
            incomingForwardEdges.push_back(e);
        } else if(e->src->getType() == "Sync"){
            incomingBackwardEdges.push_back(e);
        } else {
            cout << "Unknown type " << e->src->getType() << "\n";
            assert(0);
        }
    } 
}

bool OutputNode::sendForwardMsgs(vector<Message*>& msgs){
    assert(readyToSendForward());
    
    // calulate output activation
    output = settings->activationFnc(value);
    
    Logging::log(3, "%s%d forward out: %f", m_type.c_str(), m_id, output);
    //cout << "Out: " << output << "\n"; 
    
    auto msg = new ForwardPropagationMessage();
    for(unsigned i = 0, j = 0; i < outgoingForwardEdges.size(); i++){
        assert( 0 == outgoingForwardEdges[i]->msgStatus );
        auto msg = new ForwardPropagationMessage();
        msg->src = m_id;
        msg->dst = outgoingForwardEdges[i]->dst->getId();
        msg->activation = output;
        msgs.push_back(msg);
    }
    
    //send(msgs,outgoingForwardEdges);
    
    // reset state
    value = 0;
    forwardSeenCount = 0;
}

bool OutputNode::sendBackwardMsgs(vector<Message*>& msgs){
    assert(readyToSendBackward());   
    
    Logging::log(3, "%s%d backward: (out) %f (targ) %f", m_type.c_str(), m_id, output, target);
    
    auto delta = (target-output)*settings->deltaActivationFnc(output);
    for(unsigned i = 0; i < outgoingBackwardEdges.size(); i++){
        assert( 0 == outgoingBackwardEdges[i]->msgStatus );
        auto msg = new BackwardPropagationMessage();
        msg->src = m_id;
        msg->dst = outgoingBackwardEdges[i]->dst->getId();
        msg->delta = delta; 
        msgs.push_back(msg);
    }
    
    //send(msgs,outgoingBackwardEdges);
    
    backwardSeenCount = 0;
}

void OutputNode::onRecv(ForwardPropagationMessage* msg) {
    value += msg->activation;
    forwardSeenCount++;
    
    delete msg;
}

void OutputNode::onRecv(BackwardPropagationMessage* msg) {
    target = msg->target;
    backwardSeenCount++;
    
    delete msg;
}
