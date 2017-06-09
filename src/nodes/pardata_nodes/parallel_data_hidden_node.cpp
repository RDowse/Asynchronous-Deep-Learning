
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
    mat = activation[stateIndex]*saved_weights[updateRef[stateIndex]%context->numModels].transpose();
        
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
    deltaWeights = (context->lr/context->numModels)*(receivedDelta * activation[stateIndex])/batchSize + context->alpha*deltaWeights; // with momentum

    // Calculate next delta value
    //Eigen::VectorXf tmp = weights.transpose()*receivedDelta;
    Eigen::VectorXf tmp = saved_weights[updateRef[stateIndex]%context->numModels].transpose()*receivedDelta;
    Eigen::VectorXf delta2 = tmp.array() * activation[stateIndex].unaryExpr(context->deltaActivationFnc).array();
    
    // update step
    weights -= deltaWeights;
    updateCount++;
    saved_weights[updateCount%context->numModels] = weights;

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

    // track update reference
    if(forwardSeenCount[msg->batchIndex] == 0) updateRef[msg->batchIndex] = msg->updateNumber;
    //assert(updateRef[msg->batchIndex] == msg->updateNumber); // messed up by bias node
    
    input[msg->batchIndex] += msg->activation;
    forwardSeenCount[msg->batchIndex]++;    
    dataSetType = msg->dataSetType;
    
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