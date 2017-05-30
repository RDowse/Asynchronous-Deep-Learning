
#include "nodes/pardata_nodes/parallel_data_hidden_node.h"
#include "messages/forward_propagation_message.h"
#include "messages/backward_propagation_message.h"

std::string ParallelDataNeuralNode::HiddenNode::m_type = "Hidden";
//NodeRegister<ParallelDataNeuralNode::HiddenNode> ParallelDataNeuralNode::HiddenNode::m_reg(ParallelDataNeuralNode::HiddenNode::m_type);

void ParallelDataNeuralNode::HiddenNode::addEdge(Edge* e) {
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

bool ParallelDataNeuralNode::HiddenNode::sendForwardMsgs(vector<Message*>& msgs) {
    assert(readyToSendForward());
    
    if(!weights.size()) initWeights();
    
    // calulate output activation
    activation = input.unaryExpr(context->activationFnc);
    
    Eigen::MatrixXf mat;
    if(dataSetType == DataSetType::validation && !dropout->unset() && dropout->isEnabled())
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
            msg->time = time;
            msg->dataSetType = dataSetType;

            msg->activation = mat.col(i);
            msgs.push_back(msg);
        }
    }
    
    // reset
    input.setZero(input.size());
    forwardSeenCount = 0;
    if(dataSetType!=DataSetType::validation) swapState<BackwardTrainState<ParallelDataNeuralNode>>();
}

bool ParallelDataNeuralNode::HiddenNode::sendBackwardMsgs(vector<Message*>& msgs){
    assert(readyToSendBackward());

    // perform weight update first
    int batchSize = receivedDelta.cols();
    deltaWeights = context->lr*(receivedDelta * activation)/batchSize + context->alpha*deltaWeights; // with momentum

    newWeights -= deltaWeights; // update step  
    
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
            msg->time = time;

            msg->delta = delta2; 
            msgs.push_back(msg);
        }
    }
    
    // reset
    backwardSeenCount = 0;
    swapState<ForwardTrainState<ParallelDataNeuralNode>>();
}

void ParallelDataNeuralNode::HiddenNode::onRecv(ForwardPropagationMessage* msg) {
    if(input.size() != msg->activation.size()) input = Eigen::VectorXf::Zero(msg->activation.size());
    input += msg->activation;
    forwardSeenCount++;    
    dataSetType = msg->dataSetType;
    
    if(dataSetType==DataSetType::training) dropout->setEnabled(true);
    else dropout->setEnabled(false);
    
    if(!dropout->unset() && msg->time > time){
        dropout->nextStep(msg->time);
        time = msg->time;
    }
    
    forwardMessagePool->returnMessage(msg);
    
    // weight update step
    if(readyToSendForward())
        weights = newWeights;
}

void ParallelDataNeuralNode::HiddenNode::onRecv(BackwardPropagationMessage* msg) {
    assert(!dropout->isEnabled() || dropout->isActive());
    if(receivedDelta.cols() != msg->delta.size()) receivedDelta = Eigen::MatrixXf::Zero(weights.size(),msg->delta.size());
    int index = dstWeightIndex[msg->src];
    
    receivedDelta.row(index) = msg->delta;
    backwardSeenCount++;
    
    backwardMessagePool->returnMessage(msg);
}