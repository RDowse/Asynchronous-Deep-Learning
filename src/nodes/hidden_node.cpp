
#include "nodes/hidden_node.h"
#include "messages/forward_propagation_message.h"
#include "messages/backward_propagation_message.h"

std::string HiddenNode::m_type = "Hidden";
NodeRegister<HiddenNode> HiddenNode::m_reg(HiddenNode::m_type);

bool HiddenNode::dispatchForwardMsgs() {
    assert(readyToSend());
    
    // calc output
    output = math::activationTan(value); 
    // send messages
    for(unsigned i=0; i < forwardEdges.size(); i++){
        assert( 0 == forwardEdges[i]->msgStatus );
        auto msg = make_shared<ForwardPropagationMessage>();
        msg->value = weights[i]*output;
        forwardEdges[i]->msg = msg; // Copy message into channel
        forwardEdges[i]->msgStatus = 
            static_cast<Edge::MessageStatus>(1 + forwardEdges[i]->getDelay()); // How long until it is ready?
    }
    
    // reset
    value = 0;
    forwardSeenCount = 0;
    return true;
}

bool HiddenNode::dispatchBackwardMsgs(){
    assert(readyToSend());
    
    // perform weight update
    float delta_sum = 0;
    while(!deltas.empty()){
        // take delta values from stack, matching to weights
        int src = deltas.top().first, index = idIndexMap[src];
        float delta = deltas.top().second;
        deltas.pop();
        
        // calc delta sum
        delta_sum += delta*weights[index];
        
        // update new weights
        deltaWeights[index] = -m_graph->lr*delta*output + m_graph->alpha*deltaWeights[index];
        newWeights[index] += deltaWeights[index]; // update step  
    }
    
    // dispatch msgs, calculating delta for next nodes
    auto delta = delta_sum*output*(1-output);
    for(unsigned i = 0; i < backwardEdges.size(); i++){
        assert( 0 == backwardEdges[i]->msgStatus );
        auto msg = make_shared<BackwardPropagationMessage>();
        msg->delta = delta; 
        msg->src = m_id;
        outgoingEdges[i]->msg = msg;
        outgoingEdges[i]->msgStatus = 
            static_cast<Edge::MessageStatus>(1 + backwardEdges[i]->getDelay());
    }
    
    // reset
    backwardSeenCount = 0;
}

void HiddenNode::onRecv(shared_ptr<ForwardPropagationMessage> msg) {
    value += msg->value;
    forwardSeenCount++;
    
    // weight update step
    if(forwardSeenCount == forwardEdges.size() && m_graph->update){
        weights = newWeights;
//        cout << "id " << m_id << " , ";
//        for(auto w: weights) cout << w << " ";
//        cout << endl;
    }
}

void HiddenNode::onRecv(shared_ptr<BackwardPropagationMessage> msg) {
    // store delta value with their source node information
    deltas.push(pair<int,float>(msg->src,msg->delta));
    backwardSeenCount++;
}