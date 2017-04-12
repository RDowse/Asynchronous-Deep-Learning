/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

#include "nodes/dnn_node.h"
#include "messages/forward_propagation_message.h"
#include "messages/backward_propagation_message.h"
#include "tools/math.h"

std::string DNNNode::m_type = "DNN";
NodeRegister<DNNNode> DNNNode::m_reg(DNNNode::m_type);

bool DNNNode::onSend(shared_ptr<ForwardPropagationMessage> msg) {
    assert(readyToSend());
    // calc output
    cout << "DNN " << m_id <<": "<< value << endl;
    output = math::activation(value); 
    vector<shared_ptr<ForwardPropagationMessage>> msgs;
    msgs.reserve(forwardEdges.size());
    // send messages
    for(unsigned i=0; i < forwardEdges.size(); i++){
        msgs.push_back(make_shared<ForwardPropagationMessage>());
        assert( 0 == forwardEdges[i]->msgStatus );
        msgs[i]->value = weights[i]*output;
        forwardEdges[i]->msg = msgs[i]; // Copy message into channel
        forwardEdges[i]->msgStatus = 
            static_cast<Edge::MessageStatus>(1 + forwardEdges[i]->getDelay()); // How long until it is ready?
    }
    // reset
    value = 0;
    seenCountForward = 0;
    return true;
}

void DNNNode::onRecv(shared_ptr<ForwardPropagationMessage> msg) {
    value += msg->value;
    seenCountForward++;
}

bool DNNNode::onSend(shared_ptr<BackwardPropagationMessage> msg) {
    assert(readyToSend());
    float delta_sum = 0;
    // perform weight update first
    while(!deltas.empty()){
        // computer delta sum for next layer
        auto pair = deltas.top(); 
        float delta = pair.second;
        int src = pair.first;
        int index = idIndexMap[src];
        delta_sum += delta*weights[index];
        // update new weights
        newWeights[index] = weights[index] - m_graph->lr*(-delta*output); // update step        
    }
        
    vector<shared_ptr<BackwardPropagationMessage>> msgs;
    msgs.reserve(backwardEdges.size());
    // calculate delta for next nodes
    auto delta = delta_sum*output*(1-output);
    for(unsigned i=0; i < backwardEdges.size(); i++){
        msgs.push_back(make_shared<BackwardPropagationMessage>());
        msgs[i]->delta = delta; 
        msgs[i]->src = this->getId();
        outgoingEdges[i]->msg = msgs[i];
        assert( 0 == backwardEdges[i]->msgStatus );
        outgoingEdges[i]->msgStatus = 
            static_cast<Edge::MessageStatus>(1 + backwardEdges[i]->getDelay());
    }
}

void DNNNode::onRecv(shared_ptr<BackwardPropagationMessage> msg) {
    deltas.push(pair<int,float>(msg->src,msg->delta));
}
