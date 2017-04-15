
#include "nodes/input_node.h"
#include "messages/forward_propagation_message.h"
#include "messages/backward_propagation_message.h"
#include "tools/math.h"

std::string InputNode::m_type = "Input";
NodeRegister<InputNode> InputNode::m_reg(InputNode::m_type);

bool InputNode::onSend(shared_ptr<ForwardPropagationMessage> msg) {
    assert(readyToSend());
    assert(weights.size()==forwardEdges.size());
    
    Logging::log(3, "%s node %d input: %f", m_type.c_str(), m_id, input);
    vector<shared_ptr<ForwardPropagationMessage>> msgs;
    msgs.reserve(forwardEdges.size());
    
    for(unsigned i=0; i < forwardEdges.size(); i++){
        msgs.push_back(make_shared<ForwardPropagationMessage>());
        assert( 0 == forwardEdges[i]->msgStatus );
        msgs[i]->value = input*weights[i];
        forwardEdges[i]->msg = msgs[i]; // Copy message into channel
        forwardEdges[i]->msgStatus = 
            static_cast<Edge::MessageStatus>(1 + forwardEdges[i]->getDelay()); // How long until it is ready?
    }
    // reset seen count
    forwardSeenCount = 0;
}

void InputNode::onRecv(shared_ptr<ForwardPropagationMessage> msg){
    // received msg from sync node
    input = msg->value;
    forwardSeenCount++;
} 

bool InputNode::onSend(shared_ptr<BackwardPropagationMessage> msg) {
    assert(readyToSend());
    // perform weight update first
    while(!deltas.empty()){
        // match delta to the correct weight
        auto pair = deltas.top(); 
        float delta = pair.second;
        int src = pair.first;
        int index = idIndexMap[src];
        // new weight update
        newWeights[index] = weights[index] - m_graph->lr*delta*input; // update step        
    }
}

void InputNode::onRecv(shared_ptr<BackwardPropagationMessage> msg) {
    deltas.push(pair<int,float>(msg->src,msg->delta));
}