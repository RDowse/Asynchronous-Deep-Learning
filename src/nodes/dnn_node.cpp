/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

#include "nodes/dnn_node.h"
#include "messages/forward_propagation_message.h"
#include "messages/backward_propagation_message.h"

std::string DNNNode::m_type = "DNN";
NodeRegister<DNNNode> DNNNode::m_reg(DNNNode::m_type);

bool DNNNode::onSend(shared_ptr<ForwardPropagationMessage> msg) {
    vector<shared_ptr<ForwardPropagationMessage>> msgs;
    msgs.reserve(forwardEdges.size());

    for(unsigned i=0; i < forwardEdges.size(); i++){
        msgs.push_back(make_shared<ForwardPropagationMessage>());
        assert( 0 == forwardEdges[i]->msgStatus );
        forwardEdges[i]->msg = msgs[i]; // Copy message into channel
        forwardEdges[i]->msgStatus = 
            static_cast<Edge::MessageStatus>(1 + forwardEdges[i]->getDelay()); // How long until it is ready?
    }
}

bool DNNNode::onSend(shared_ptr<BackwardPropagationMessage> msg) {}

void DNNNode::onRecv(shared_ptr<ForwardPropagationMessage> msg) {}

void DNNNode::onRecv(shared_ptr<BackwardPropagationMessage> msg) {}
