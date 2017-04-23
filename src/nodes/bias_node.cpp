
#include "nodes/bias_node.h"
#include "messages/forward_propagation_message.h"
#include "messages/backward_propagation_message.h"

std::string BiasNode::m_type = "Bias";
NodeRegister<BiasNode> BiasNode::m_reg(BiasNode::m_type);

bool BiasNode::onSend(vector< shared_ptr<Message> >& msgs){
    assert(readyToSend());

    for(unsigned i=0; i < forwardEdges.size(); i++){
        assert( 0 == forwardEdges[i]->msgStatus );
        auto msg = make_shared<ForwardPropagationMessage>();
        msg->value = value*weights[i];
        forwardEdges[i]->msg = msg; // Copy message into channel
        forwardEdges[i]->msgStatus = 
            static_cast<Edge::MessageStatus>(1 + forwardEdges[i]->getDelay()); // How long until it is ready?
    }
    
    forwardSeenCount = 0;
    backwardSeenCount = 0;
}

void BiasNode::onRecv(shared_ptr<ForwardPropagationMessage> msg) {
    // notifying msg from sync node
    forwardSeenCount++;

    // weight update step
    if(forwardSeenCount == forwardEdges.size() && m_graph->update)
        weights = newWeights;
}

void BiasNode::onRecv(shared_ptr<BackwardPropagationMessage> msg) {
    // store delta value with their source node information
    deltas.push(pair<int,float>(msg->src,msg->delta));
    backwardSeenCount++;
    
    if(backwardSeenCount == forwardEdges.size()){
        // perform weight update
        assert(deltas.size() == forwardEdges.size());
        while(!deltas.empty()){
            // take delta values from stack, matching to weights
            auto pair = deltas.top(); 
            float delta = pair.second;
            int src = pair.first;
            int index = idIndexMap[src];
            deltas.pop();
            
            // update new weights
            deltaWeights[index] = -m_graph->lr*delta + m_graph->alpha*deltaWeights[index];
            newWeights[index] += deltaWeights[index]; // update step 
        }
    }
}
