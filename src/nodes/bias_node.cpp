
#include "nodes/bias_node.h"
#include "messages/forward_propagation_message.h"
#include "messages/backward_propagation_message.h"

std::string BiasNode::m_type = "Bias";
NodeRegister<BiasNode> BiasNode::m_reg(BiasNode::m_type);

bool BiasNode::onSend(shared_ptr<ForwardPropagationMessage> msg) {
    assert(readyToSend());
    assert(weights.size()==outgoingEdges.size());
    
    cout << "BIAS " << m_id <<": "<< value << endl;
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
    sent = true;
}

bool BiasNode::onSend(shared_ptr<BackwardPropagationMessage> msg) {}

void BiasNode::onRecv(shared_ptr<ForwardPropagationMessage> msg) {}

void BiasNode::onRecv(shared_ptr<BackwardPropagationMessage> msg) {}
