
#include "nodes/bias_node.h"
#include "messages/forward_propagation_message.h"
#include "messages/backward_propagation_message.h"

std::string BiasNode::m_type = "Bias";
NodeRegister<BiasNode> BiasNode::m_reg(BiasNode::m_type);

bool BiasNode::onSend(shared_ptr<ForwardPropagationMessage> msg) {}

bool BiasNode::onSend(shared_ptr<BackwardPropagationMessage> msg) {}

void BiasNode::onRecv(shared_ptr<ForwardPropagationMessage> msg) {}

void BiasNode::onRecv(shared_ptr<BackwardPropagationMessage> msg) {}
