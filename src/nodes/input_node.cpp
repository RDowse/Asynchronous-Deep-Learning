
#include "nodes/input_node.h"
#include "messages/forward_propagation_message.h"
#include "messages/backward_propagation_message.h"
#include "tools/math.h"

std::string InputNode::m_type = "Input";
NodeRegister<InputNode> InputNode::m_reg(InputNode::m_type);

bool InputNode::sendForwardMsgs(vector<shared_ptr<Message>>& msgs){
    assert(readyToSend());
    
    Logging::log(3, "%s node %d input: %f", m_type.c_str(), m_id, input);
    
    msgs.reserve(weights.size());
    for(unsigned i=0; i < weights.size(); i++){
        auto msg = make_shared<ForwardPropagationMessage>();
        msg->value = input*weights[i];
        msgs.push_back(msg);
    }
    
    send(msgs,outgoingForwardEdges);
    
    // reset 
    forwardSeenCount = 0;
}

bool InputNode::sendBackwardMsgs(vector<shared_ptr<Message>>& msgs){
    assert(readyToSend());
    // perform weight update first
    while(!deltas.empty()){
        // match delta to the correct weight
        int src = deltas.top().first, index = idIndexMap[src];
        float delta = deltas.top().second;
        deltas.pop();
        
        // new weight update
        deltaWeights[index] = -settings->lr*delta*input + settings->alpha*deltaWeights[index];
        newWeights[index] += deltaWeights[index]; // update step        
    }
    // notify sync node
    msgs.push_back(make_shared<BackwardPropagationMessage>());
    send(msgs,outgoingBackwardEdges);
    // reset
    backwardSeenCount = 0;
}

void InputNode::onRecv(shared_ptr<ForwardPropagationMessage> msg){
    input = msg->value;
    forwardSeenCount++;
    
    // weight update step
    if(forwardSeenCount == outgoingForwardEdges.size() && settings->update){
        weights = newWeights;
    }
} 

void InputNode::onRecv(shared_ptr<BackwardPropagationMessage> msg) {
    deltas.push(pair<int,float>(msg->src,msg->delta));
    backwardSeenCount++;
}