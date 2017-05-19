
#include "nodes/bias_node.h"
#include "messages/forward_propagation_message.h"
#include "messages/backward_propagation_message.h"

std::string BiasNode::m_type = "Bias";
NodeRegister<BiasNode> BiasNode::m_reg(BiasNode::m_type);

void BiasNode::addEdge(Edge* e){
    // add to original edge sets
    Node::addEdge(e);
    // check edge belongs to this node
    if(e->src->getId() == m_id){
        if(e->dst->getType() == "Sync"){
            outgoingBackwardEdges.push_back(e);
        } else if(e->dst->getType() == "Hidden"
                || e->dst->getType() == "Output"){
            outgoingForwardEdges.push_back(e);
            dstWeightIndex[e->dst->getId()] = map_index++;
        } else {
            cout << "Unknown type " << e->dst->getType() << "\n";
            assert(0);
        }
    } else if(e->dst->getId() == m_id){
        if(e->src->getType() == "Sync"){
            incomingForwardEdges.push_back(e);
        } else if(e->src->getType() == "Hidden"
                || e->src->getType() == "Output"){
            incomingBackwardEdges.push_back(e);
        } else {
            cout << "Unknown type " << e->src->getType() << "\n";
            assert(0);
        }
    } 
}

bool BiasNode::sendForwardMsgs(vector<Message*>& msgs){
    assert(readyToSendForward());
    
    if(!weights.size()) initWeights();
    
    MatrixXf mat = input*weights.transpose();
    
    msgs.reserve(outgoingForwardEdges.size());
    for(unsigned i = 0; i < outgoingForwardEdges.size(); i++){
        assert( 0 == outgoingForwardEdges[i]->msgStatus );
        auto msg = forwardMessagePool->getMessage();
        msg->src = m_id;
        msg->dst = outgoingForwardEdges[i]->dst->getId();
        msg->activation = mat.col(i);
        msgs.push_back(msg);
    }
    
    forwardSeenCount = 0;
}

bool BiasNode::sendBackwardMsgs(vector<Message*>& msgs){
    assert(readyToSendBackward());
    
    deltaWeights = context->lr*(receivedDelta * input) + context->alpha*deltaWeights;

    newWeights -= deltaWeights; // update step  
    
    msgs.reserve(outgoingBackwardEdges.size());
    for(unsigned i = 0; i < outgoingBackwardEdges.size(); i++){
        assert( 0 == outgoingBackwardEdges[i]->msgStatus );        
        auto msg = backwardMessagePool->getMessage();
        msg->src = m_id;
        msg->dst = outgoingBackwardEdges[i]->dst->getId();
        msgs.push_back(msg);
    }
    assert(msgs.size() == 1);
    
    backwardSeenCount = 0;
}


void BiasNode::onRecv(ForwardPropagationMessage* msg) {
    // notifying msg from sync node
    forwardSeenCount++;
    input = msg->activation;
    
    assert(input.sum() == input.size()); // check all values are 1
    
    forwardMessagePool->returnMessage(msg);
    
    // weights update step
    if(readyToSendForward())
        weights = newWeights;
}

void BiasNode::onRecv(BackwardPropagationMessage* msg) {  
    if(!receivedDelta.size()) receivedDelta = Eigen::MatrixXf::Zero(weights.size(),context->batchSize);
    int index = dstWeightIndex[msg->src];
    receivedDelta.row(index) = msg->delta;
    backwardSeenCount++;
    
    backwardMessagePool->returnMessage(msg);
}
