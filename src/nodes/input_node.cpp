/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

#include "nodes/input_node.h"
#include "messages/forward_propagation_message.h"
#include "messages/backward_propagation_message.h"
#include "tools/math.h"

std::string InputNode::m_type = "Input";
NodeRegister<InputNode> InputNode::m_reg(InputNode::m_type);

bool InputNode::onSend(shared_ptr<ForwardPropagationMessage> msg) {
    assert(readyToSend());
    assert(weights.size()==outgoingEdges.size());
    
    cout << "INPUT " << m_id <<": "<< input << endl;
    vector<shared_ptr<ForwardPropagationMessage>> msgs;
    msgs.reserve(outgoingEdges.size());
    
    for(unsigned i=0; i < outgoingEdges.size(); i++){
        msgs.push_back(make_shared<ForwardPropagationMessage>());
        assert( 0 == outgoingEdges[i]->msgStatus );
        msgs[i]->value = input*weights[i];
        outgoingEdges[i]->msg = msgs[i]; // Copy message into channel
        outgoingEdges[i]->msgStatus = 
            static_cast<Edge::MessageStatus>(1 + outgoingEdges[i]->getDelay()); // How long until it is ready?
    }
    sent = true;
}

void InputNode::onRecv(shared_ptr<ForwardPropagationMessage> msg){
    input = msg->value;
    // sending = true ?
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
    // trigger forward propagation
    sent = false;
}

void InputNode::onRecv(shared_ptr<BackwardPropagationMessage> msg) {
    deltas.push(pair<int,float>(msg->src,msg->delta));
}