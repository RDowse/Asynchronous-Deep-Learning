
#include "nodes/output_node.h"

#include "messages/forward_propagation_message.h"
#include "messages/backward_propagation_message.h"
#include "tools/math.h"

std::string OutputNode::m_type = "Output";
NodeRegister<OutputNode> OutputNode::m_reg(OutputNode::m_type);

bool OutputNode::onSend(shared_ptr<ForwardPropagationMessage> msg) {
    assert(readyToSend());
    output = math::activation(value);
    seenCount = 0;
    value = 0;
    
    cout << "OUTPUT " << m_id <<": "<< value << endl; 
    cout << "OUTPUT O " << m_id <<": "<< output << endl; 

}

void OutputNode::onRecv(shared_ptr<ForwardPropagationMessage> msg) {
    value += msg->value;
    seenCount++;
}

bool OutputNode::onSend(shared_ptr<BackwardPropagationMessage> msg) {
    assert(readyToSend());   
    vector<shared_ptr<BackwardPropagationMessage>> msgs;
    msgs.reserve(outgoingEdges.size());
    auto target = 0.0;
    auto delta = -(target-output)*output*(1-output);
    for(int i = 0; i <= outgoingEdges.size(); ++i){
        msgs.push_back(make_shared<BackwardPropagationMessage>());
        msgs[i]->delta = delta; 
        msgs[i]->src = this->getId();
        outgoingEdges[i]->msg = msgs[i];
        assert( 0 == outgoingEdges[i]->msgStatus );
        outgoingEdges[i]->msgStatus = 
            static_cast<Edge::MessageStatus>(1 + outgoingEdges[i]->getDelay());
    }   
}

void OutputNode::onRecv(shared_ptr<BackwardPropagationMessage> msg) {
    assert(0); // should never occur
}
