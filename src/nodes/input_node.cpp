
#include "nodes/input_node.h"
#include "messages/forward_propagation_message.h"
#include "messages/backward_propagation_message.h"
#include "tools/math.h"

std::string InputNode::m_type = "Input";
NodeRegister<InputNode> InputNode::m_reg(InputNode::m_type);

bool InputNode::dispatchForwardMsgs(){
    assert(readyToSend());
    
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
    // reset 
    forwardSeenCount = 0;
}

bool InputNode::dispatchBackwardMsgs(){
    assert(readyToSend());
    // perform weight update first
    while(!deltas.empty()){
        // match delta to the correct weight
        int src = deltas.top().first, index = idIndexMap[src];
        float delta = deltas.top().second;
        deltas.pop();
        
        // new weight update
        deltaWeights[index] = -m_graph->lr*delta*input + m_graph->alpha*deltaWeights[index];
        newWeights[index] += deltaWeights[index]; // update step        
    }
    // notify sync node
    backwardSyncEdge->msg = make_shared<BackwardPropagationMessage>();
    backwardSyncEdge->msgStatus = 
            static_cast<Edge::MessageStatus>(1 + backwardSyncEdge->getDelay());
    // reset
    backwardSeenCount = 0;
}

void InputNode::onRecv(shared_ptr<ForwardPropagationMessage> msg){
    input = msg->value;
    forwardSeenCount++;
    
    // weight update step
    if(forwardSeenCount == forwardEdges.size() && m_graph->update){
        weights = newWeights;
    }
} 

void InputNode::onRecv(shared_ptr<BackwardPropagationMessage> msg) {
    deltas.push(pair<int,float>(msg->src,msg->delta));
    backwardSeenCount++;
}