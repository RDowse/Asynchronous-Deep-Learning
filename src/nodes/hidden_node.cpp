
#include "nodes/hidden_node.h"
#include "messages/forward_propagation_message.h"
#include "messages/backward_propagation_message.h"

std::string NeuralNode::HiddenNode::m_type = "Hidden";
NodeRegister<NeuralNode::HiddenNode> NeuralNode::HiddenNode::m_reg(NeuralNode::HiddenNode::m_type);

void NeuralNode::HiddenNode::addEdge(Edge* e) {
    // add to original edge sets
    Node::addEdge(e);
    if(e->src->getId() == m_id){
        if(e->dst->getId() > m_id){ // change based on type of edge.
            outgoingForwardEdges.push_back(e);
            dstWeightIndex[e->dst->getId()] = map_index++;
        } else {
            outgoingBackwardEdges.push_back(e);
        }
    } else if(e->dst->getId() == m_id){
        if(e->src->getId() < m_id){ // change based on type of edge.
            incomingForwardEdges.push_back(e);
        } else {
            incomingBackwardEdges.push_back(e);
        }
    }
}

bool NeuralNode::HiddenNode::sendForwardMsgs(vector<Message*>& msgs) {
    assert(readyToSendForward());
    
    if(!weights.size()) initWeights();
    
    // calulate output activation
    activation = input.unaryExpr(context->activationFnc);
    
    MatrixXf mat = activation*weights.transpose();
    
    msgs.reserve(outgoingForwardEdges.size());
    assert(weights.size() == outgoingForwardEdges.size());
    for(unsigned i = 0; i < outgoingForwardEdges.size(); i++){
        assert( 0 == outgoingForwardEdges[i]->msgStatus );
        auto msg = forwardMessagePool->getMessage();
        msg->src = m_id;
        msg->dst = outgoingForwardEdges[i]->dst->getId();
        msg->activation = mat.col(i);
        msgs.push_back(msg);
    }
    
    // reset
    input.setZero(input.size());
    forwardSeenCount = 0;
}

bool NeuralNode::HiddenNode::sendBackwardMsgs(vector<Message*>& msgs){
    assert(readyToSendBackward());

    // perform weight update first
    deltaWeights = context->lr*(receivedDelta * activation) + context->alpha*deltaWeights; // with momentum

    newWeights -= deltaWeights; // update step  
    
    // Calculate next delta value
    Eigen::VectorXf tmp = weights.transpose()*receivedDelta;
    Eigen::VectorXf delta2 = tmp.array()*activation.unaryExpr(context->deltaActivationFnc).array();
    
    msgs.reserve(outgoingBackwardEdges.size());
    for(unsigned i = 0; i < outgoingBackwardEdges.size(); i++){
        assert( 0 == outgoingBackwardEdges[i]->msgStatus );
        auto msg = backwardMessagePool->getMessage();
        msg->src = m_id;
        msg->dst = outgoingBackwardEdges[i]->dst->getId();
        
        msg->delta = delta2; 
        msgs.push_back(msg);
    }
    
    // reset
    backwardSeenCount = 0;
}

void NeuralNode::HiddenNode::onRecv(ForwardPropagationMessage* msg) {
    if(!input.size()) input = Eigen::VectorXf::Zero(msg->activation.size());
    input += msg->activation;
    forwardSeenCount++;
    
    forwardMessagePool->returnMessage(msg);
    
    // weight update step
    if(readyToSendForward())
        weights = newWeights;
}

void NeuralNode::HiddenNode::onRecv(BackwardPropagationMessage* msg) {
    if(!receivedDelta.size()) receivedDelta = Eigen::MatrixXf::Zero(weights.size(),context->batchSize);
    int index = dstWeightIndex[msg->src];
    receivedDelta.row(index) = msg->delta;
    backwardSeenCount++;
    
    backwardMessagePool->returnMessage(msg);
}