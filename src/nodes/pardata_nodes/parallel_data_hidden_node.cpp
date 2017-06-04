
#include "nodes/pardata_nodes/parallel_data_hidden_node.h"
#include "messages/forward_propagation_message.h"
#include "messages/backward_propagation_message.h"

std::string ParallelDataNeuralNode::HiddenNode::m_type = "Hidden";

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

bool ParallelDataNeuralNode::HiddenNode::sendForwardMsgs(vector<Message*>& msgs, int stateIndex) {
    assert(readyToSendForward(stateIndex));
    
    if(!weights.size()) initWeights();
    
    // calulate output activation
    activation[stateIndex] = input[stateIndex].unaryExpr(context->activationFnc);
    
    Eigen::MatrixXf mat;
    if(dataSetType == DataSetType::validating && !dropout->unset() && dropout->isEnabled())
        mat = 0.5*activation[stateIndex]*weights.transpose(); // for dropout based on probability, TODO correct for prime (adjustable probability)
    else 
        mat = activation[stateIndex]*weights.transpose();
        
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
            msg->batchIndex = stateIndex;

            msg->activation = mat.col(i);
            msgs.push_back(msg);
        }
    }
    
    // store current weight values
    prevWeights[stateIndex] = weights;
    
    // reset
    input[stateIndex].setZero(input[stateIndex].size());
    forwardSeenCount[stateIndex] = 0;
    
    delete state[stateIndex];
    state[stateIndex] = new BackwardTrainState<ParallelDataNeuralNode>();
}

bool ParallelDataNeuralNode::HiddenNode::sendBackwardMsgs(vector<Message*>& msgs, int stateIndex){
    assert(readyToSendBackward(stateIndex));

    // perform weight update first
    int batchSize = receivedDelta.cols();
    deltaWeights = context->lr*(receivedDelta * activation[stateIndex])/batchSize + context->alpha*deltaWeights; // with momentum

    // Calculate next delta value
    Eigen::VectorXf tmp = weights.transpose()*receivedDelta;
    //Eigen::VectorXf tmp = prevWeights[stateIndex].transpose()*receivedDelta;
    Eigen::VectorXf delta2 = tmp.array() * activation[stateIndex].unaryExpr(context->deltaActivationFnc).array();
    
    // update step  
    weights -= deltaWeights;

    msgs.reserve(outgoingBackwardEdges.size());
    for(unsigned i = 0; i < outgoingBackwardEdges.size(); i++){
        if(dropout->isPrevLayerNodeActive(i)){
            assert( 0 == outgoingBackwardEdges[i]->msgStatus );
            auto msg = backwardMessagePool->getMessage();
            msg->src = m_id;
            msg->dst = outgoingBackwardEdges[i]->dst->getId();
            msg->time = time;
            msg->batchIndex = stateIndex;

            msg->delta = delta2; 
            msgs.push_back(msg);
        }
    }
    
    // reset
    backwardSeenCount[stateIndex] = 0;
    
    delete state[stateIndex];
    state[stateIndex] = new ForwardTrainState<ParallelDataNeuralNode>();
}

void ParallelDataNeuralNode::HiddenNode::onRecv(ForwardPropagationMessage* msg) {
    if(input[msg->batchIndex].size() != msg->activation.size()) 
        input[msg->batchIndex] = Eigen::VectorXf::Zero(msg->activation.size());

    input[msg->batchIndex] += msg->activation;
    forwardSeenCount[msg->batchIndex]++;    
    dataSetType = msg->dataSetType;
    
    if(dataSetType==DataSetType::training) dropout->setEnabled(true);
    else dropout->setEnabled(false);
    
    if(!dropout->unset() && msg->time > time){
        dropout->nextStep(msg->time);
        time = msg->time;
    }
    
    forwardMessagePool->returnMessage(msg);
}

void ParallelDataNeuralNode::HiddenNode::onRecv(BackwardPropagationMessage* msg) {
    assert(!dropout->isEnabled() || dropout->isActive());
    if(receivedDelta.cols() != msg->delta.size()) receivedDelta = Eigen::MatrixXf::Zero(weights.size(),msg->delta.size());
    int index = dstWeightIndex[msg->src];
    
    receivedDelta.row(index) = msg->delta;
    backwardSeenCount[msg->batchIndex]++;
    
    backwardMessagePool->returnMessage(msg);
}