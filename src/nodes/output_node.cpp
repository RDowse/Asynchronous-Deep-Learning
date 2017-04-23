
#include "nodes/output_node.h"

#include "messages/forward_propagation_message.h"
#include "messages/backward_propagation_message.h"
#include "tools/math.h"

std::string OutputNode::m_type = "Output";
NodeRegister<OutputNode> OutputNode::m_reg(OutputNode::m_type);

bool OutputNode::dispatchForwardMsgs(vector<shared_ptr<Message>>& msgs){
    assert(readyToSend());
    
    // calculate output for the node
    output = math::activationTan(value);
    Logging::log(3, "%s node %d output: %f", m_type.c_str(), m_id, output);
    // prepare the forward message
    auto msg = make_shared<ForwardPropagationMessage>();
    msg->value = output;
    msg->src = m_id;
    
    // send the output to the sync node
    syncEdge->msg = msg; 
    syncEdge->msgStatus = 
        static_cast<Edge::MessageStatus>(1 + syncEdge->getDelay()); 
    
    // reset state
    forwardSeenCount = 0;
    value = 0;
}

bool OutputNode::dispatchBackwardMsgs(vector<shared_ptr<Message>>& msgs){
    assert(readyToSend());   
    
    cout << m_type.c_str() << " " << m_id << ": (out)" << output << ", (targ)" << target << endl;
    
    auto delta = -(target-output)*output*(1-output);
    
    for(int i = 0; i < backwardEdges.size(); ++i){
        auto msg = make_shared<BackwardPropagationMessage>();
        msg->delta = delta; 
        msg->src = m_id;
        
        assert( 0 == backwardEdges[i]->msgStatus );
        backwardEdges[i]->msg = msg;
        backwardEdges[i]->msgStatus = 
            static_cast<Edge::MessageStatus>(1 + outgoingEdges[i]->getDelay());
    }   
    
    backwardSeenCount = 0;
}

void OutputNode::onRecv(shared_ptr<ForwardPropagationMessage> msg) {
    value += msg->value;
    forwardSeenCount++;
}

void OutputNode::onRecv(shared_ptr<BackwardPropagationMessage> msg) {
    target = msg->target;
    backwardSeenCount++;
}
