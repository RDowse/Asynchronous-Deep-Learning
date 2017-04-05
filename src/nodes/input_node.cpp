/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

#include "nodes/input_node.h"
#include "messages/forward_propagation_message.h"

std::string InputNode::m_type = "Input";
NodeRegister<InputNode> InputNode::m_reg(InputNode::m_type);

bool InputNode::onSend(shared_ptr<ForwardPropagationMessage> msg) {
    assert(readyToSend());
    
    vector<shared_ptr<ForwardPropagationMessage>> msgs;
    msgs.reserve(outgoingEdges.size());

    for(unsigned i=0; i < outgoingEdges.size(); i++){
        msgs.push_back(make_shared<ForwardPropagationMessage>());
        assert( 0 == outgoingEdges[i]->msgStatus );
        msgs[i]->value = value*weights[i];
        outgoingEdges[i]->msg = msgs[i]; // Copy message into channel
        outgoingEdges[i]->msgStatus = 
            static_cast<Edge::MessageStatus>(1 + outgoingEdges[i]->getDelay()); // How long until it is ready?
    }
}

void InputNode::onRecv(shared_ptr<ForwardPropagationMessage> msg){
    value = msg->value;
} 