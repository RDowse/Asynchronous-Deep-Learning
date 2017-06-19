
#include "nodes/hidden_node.h"
#include "messages/forward_propagation_message.h"
#include "messages/backward_propagation_message.h"

std::string NeuralNode::HiddenNode::m_type = "Hidden";

void NeuralNode::HiddenNode::addEdge(Edge* e) {
    // add to original edge sets
    Node::addEdge(e);
    if(e->src->getId() == id){
        if(e->dst->getId() > id){ // change based on type of edge.
            outgoingForwardEdges.push_back(e);
            dstWeightIndex[e->dst->getId()] = map_index++;
        } else {
            outgoingBackwardEdges.push_back(e);
        }
    } else if(e->dst->getId() == id){
        if(e->src->getId() < id){ // change based on type of edge.
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
    
    Eigen::MatrixXf mat;
    if(dataSetType != DataSetType::training && !dropout->unset() && dropout->isEnabled())
        mat = 0.5*activation*weights.transpose(); // for dropout based on probability, TODO correct for prime (adjustable probability)
    else 
        mat = activation*weights.transpose();
        
    msgs.reserve(outgoingForwardEdges.size());
    assert(weights.size() == outgoingForwardEdges.size());
    for(unsigned i = 0; i < outgoingForwardEdges.size(); i++){
        if(dropout->isNextLayerNodeActive(i) || dataSetType != DataSetType::training){
            assert( 0 == outgoingForwardEdges[i]->msgStatus );
            auto msg = forwardMessagePool->getMessage();
            msg->src = id;
            msg->dst = outgoingForwardEdges[i]->dst->getId();
            msg->batchNum = batchNum;
            msg->dataSetType = dataSetType;
            
            //if(context->epoch==context->maxEpoch-1) context->insertHist(mat.col(i));
            
            msg->activation = mat.col(i);
            msgs.push_back(msg);
        }
    }
    
    // reset
    input.setZero(input.size());
    forwardSeenCount = 0;
    if(dataSetType==DataSetType::training) swapState<BackwardTrainState<NeuralNode>>();
}

bool NeuralNode::HiddenNode::sendBackwardMsgs(vector<Message*>& msgs){
    assert(readyToSendBackward());

    // perform weight update first
    int batchSize = receivedDelta.cols();
    deltaWeights = context->lr*(receivedDelta * activation)/batchSize + context->alpha*deltaWeights; // with momentum
    
    // Calculate next delta value
    Eigen::VectorXf tmp = weights.transpose()*receivedDelta;
    Eigen::VectorXf delta2 = tmp.array() * activation.unaryExpr(context->deltaActivationFnc).array();

    weights -= deltaWeights; // update step  
    weights = context->regularizationFnc(weights, context->c);

    msgs.reserve(outgoingBackwardEdges.size());
    for(unsigned i = 0; i < outgoingBackwardEdges.size(); i++){
        if(dropout->isPrevLayerNodeActive(i)){
            assert( 0 == outgoingBackwardEdges[i]->msgStatus );
            auto msg = backwardMessagePool->getMessage();
            msg->src = id;
            msg->dst = outgoingBackwardEdges[i]->dst->getId();
            msg->batchNum = batchNum;

            msg->delta = delta2; 
            msgs.push_back(msg);
        }
    }
    
    // reset
    backwardSeenCount = 0;
    swapState<ForwardTrainState<NeuralNode>>();
}

void NeuralNode::HiddenNode::onRecv(ForwardPropagationMessage* msg) {
    if(input.size() != msg->activation.size()) input = Eigen::VectorXf::Zero(msg->activation.size());
    input += msg->activation;
    forwardSeenCount++;    
    dataSetType = msg->dataSetType;
    
    if(dataSetType==DataSetType::training) dropout->setEnabled(true);
    else dropout->setEnabled(false);
    
    if(!dropout->unset() && msg->batchNum > batchNum){
        dropout->nextStep(msg->batchNum);
        batchNum = msg->batchNum;
    }
    
    forwardMessagePool->returnMessage(msg);
}

void NeuralNode::HiddenNode::onRecv(BackwardPropagationMessage* msg) {
    assert(!dropout->isEnabled() || dropout->isActive());
    if(receivedDelta.cols() != msg->delta.size()) receivedDelta = Eigen::MatrixXf::Zero(weights.size(),msg->delta.size());
    int index = dstWeightIndex[msg->src];
    
    receivedDelta.row(index) = msg->delta;
    backwardSeenCount++;
    
    backwardMessagePool->returnMessage(msg);
}