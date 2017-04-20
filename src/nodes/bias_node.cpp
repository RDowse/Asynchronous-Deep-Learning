
#include "nodes/bias_node.h"
#include "messages/forward_propagation_message.h"
#include "messages/backward_propagation_message.h"

std::string BiasNode::m_type = "Bias";
NodeRegister<BiasNode> BiasNode::m_reg(BiasNode::m_type);

bool BiasNode::dispatchMsgs(){
    assert(readyToSend());

    for(unsigned i=0; i < forwardEdges.size(); i++){
        assert( 0 == forwardEdges[i]->msgStatus );
        auto msg = make_shared<ForwardPropagationMessage>();
        msg->value = value*weights[i];
        forwardEdges[i]->msg = msg; // Copy message into channel
        forwardEdges[i]->msgStatus = 
            static_cast<Edge::MessageStatus>(1 + forwardEdges[i]->getDelay()); // How long until it is ready?
    }
    
    seenCount=0;
}

void BiasNode::onRecv(shared_ptr<ForwardPropagationMessage> msg) {
    // notifying msg from sync node
    seenCount++;
}

void BiasNode::onRecv(shared_ptr<BackwardPropagationMessage> msg) {}
