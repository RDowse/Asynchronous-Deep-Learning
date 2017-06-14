
#include "nodes/async_nodes/async_hidden_node.h"
#include "messages/forward_propagation_message.h"
#include "messages/backward_propagation_message.h"

std::string AsyncNeuralNode::HiddenNode::m_type = "Hidden";

void AsyncNeuralNode::HiddenNode::addEdge(Edge* e) {
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

bool AsyncNeuralNode::HiddenNode::sendForwardMsgs(vector<Message*>& msgs) {
    
    if(!weights.size()) initWeights();
    
    input *= float(incomingForwardEdges.size()/forwardSeenCount);
    
    // calulate output activation
    activation = input.unaryExpr(context->activationFnc);
    
    //Eigen::MatrixXf mat;
    //mat = activation*weights.transpose();
    
    Eigen::MatrixXf mat;
    if(dataSetType != DataSetType::training && !dropout->unset() && dropout->isEnabled())
        mat = 0.5*activation*weights.transpose(); // for dropout based on probability, TODO correct for prime (adjustable probability)
    else 
        mat = activation*weights.transpose();
        
    msgs.reserve(outgoingForwardEdges.size());
    assert(weights.size() == outgoingForwardEdges.size());
    for(unsigned i = 0; i < outgoingForwardEdges.size(); i++){
        if(dropout->isNextLayerNodeActive(i)){
            assert( 0 == outgoingForwardEdges[i]->msgStatus );
            auto msg = forwardMessagePool->getMessage();
            msg->src = m_id;
            msg->dst = outgoingForwardEdges[i]->dst->getId();
            msg->time = curr_forward_batch;
            msg->dataSetType = dataSetType;

            msg->activation = mat.col(i);
            msgs.push_back(msg);
            
            numMessagesSentForward++;
        }
    }
    
    curr_forward_batch++;
    receivedFirstMessage = false;
    
    // reset
    forwardSeenCount = 0;
    input.setZero(input.size());
    if(dataSetType == DataSetType::training) swapState<BackwardTrainState<AsyncNeuralNode>>();
}

bool AsyncNeuralNode::HiddenNode::sendBackwardMsgs(vector<Message*>& msgs){
    
    int batchSize = receivedDelta.cols();
    deltaWeights = context->lr*(receivedDelta * activation)/batchSize + context->alpha*deltaWeights; // with momentum

    weights -= deltaWeights; // update step 
    
    // Calculate next delta value
    Eigen::VectorXf tmp = weights.transpose()*receivedDelta;
    Eigen::VectorXf delta2 = tmp.array() * activation.unaryExpr(context->deltaActivationFnc).array();

    msgs.reserve(outgoingBackwardEdges.size());
    for(unsigned i = 0; i < outgoingBackwardEdges.size(); i++){
        if(dropout->isPrevLayerNodeActive(i)){
            assert( 0 == outgoingBackwardEdges[i]->msgStatus );
            auto msg = backwardMessagePool->getMessage();
            msg->src = m_id;
            msg->dst = outgoingBackwardEdges[i]->dst->getId();
            msg->time = curr_backward_batch;

            msg->delta = delta2; 
            msgs.push_back(msg);
            
            numMessagesSentBackward++;
        }
    }
    
    curr_backward_batch++;
    receivedFirstMessage = false;
    
    // reset delta values
    receivedDelta.Zero(receivedDelta.rows(),receivedDelta.cols());
    
    backwardSeenCount = 0;
    swapState<ForwardTrainState<AsyncNeuralNode>>();
}

void AsyncNeuralNode::HiddenNode::onRecv(ForwardPropagationMessage* msg) {
    if(input.size() != msg->activation.size()) input = Eigen::VectorXf::Zero(msg->activation.size());
    
    if(forwardDiscardMsgCheck(msg)) return;
            
    forwardSeenCount++;    
    input += msg->activation;
    dataSetType = msg->dataSetType;
    
    if(dataSetType==DataSetType::training) dropout->setEnabled(true);
    else dropout->setEnabled(false);
    
    if(!dropout->unset() && msg->time > time){
        dropout->nextStep(msg->time);
        time = msg->time;
    }
    
    forwardMessagePool->returnMessage(msg);
}

void AsyncNeuralNode::HiddenNode::onRecv(BackwardPropagationMessage* msg) {
    assert(!dropout->isEnabled() || dropout->isActive());
    if(receivedDelta.cols() != msg->delta.size()) receivedDelta = Eigen::MatrixXf::Zero(weights.size(),msg->delta.size());

    if(backwardDiscardMsgCheck(msg)) return;
            
    int index = dstWeightIndex[msg->src];
    receivedDelta.row(index) = msg->delta;
    backwardSeenCount++;
    
    backwardMessagePool->returnMessage(msg);
}