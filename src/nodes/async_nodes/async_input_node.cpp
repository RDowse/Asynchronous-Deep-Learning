
#include "nodes/async_nodes/async_input_node.h"
#include "messages/forward_propagation_message.h"
#include "messages/backward_propagation_message.h"
#include "tools/math.h"

std::string AsyncNeuralNode::InputNode::m_type = "Input";

void AsyncNeuralNode::InputNode::addEdge(Edge* e) {
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
             cout << "Unknown type " << e->dst->getType() << "\n";
             assert(0);
         }
     } 
     
     // tmp tests
     assert(outgoingBackwardEdges.size() <= 1);        
     assert(incomingForwardEdges.size() <= 1);
 }

bool AsyncNeuralNode::InputNode::sendForwardMsgs(vector<Message*>& msgs){
    assert(readyToSendForward());
    
    if(!weights.size()) initWeights();
    
    Eigen::MatrixXf mat = activation*weights.transpose();
    
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
            
            numMessagesSent++;
        }
    }
    
    curr_forward_batch++;
    
    forwardSeenCount = 0;
    
    // reset 
    if(dataSetType == DataSetType::training) swapState<BackwardTrainState<AsyncNeuralNode>>();
}

bool AsyncNeuralNode::InputNode::sendBackwardMsgs(vector<Message*>& msgs){
    
    // perform weight update first
    int batchSize = receivedDelta.cols();
    deltaWeights = context->lr*(receivedDelta * activation)/batchSize + context->alpha*deltaWeights;
    
    weights -= deltaWeights;
    
    msgs.reserve(outgoingBackwardEdges.size());
    for(unsigned i = 0; i < outgoingBackwardEdges.size(); i++){
        assert( 0 == outgoingBackwardEdges[i]->msgStatus );
        auto msg = backwardMessagePool->getMessage();
        msg->src = m_id;
        msg->dst = outgoingBackwardEdges[i]->dst->getId();
        msg->time = curr_forward_batch;

        msgs.push_back(msg);
        
        numMessagesSent++;
    }
    
    curr_backward_batch++;
    receivedFirstMessage = false;
    
    // reset delta values
    receivedDelta.Zero(receivedDelta.rows(),receivedDelta.cols());
    
    backwardSeenCount = 0;
    swapState<ForwardTrainState<AsyncNeuralNode>>();
}

void AsyncNeuralNode::InputNode::onRecv(ForwardPropagationMessage* msg){
    activation = msg->activation;
    forwardSeenCount++;
    
    dataSetType = msg->dataSetType;
    
    if(dataSetType==DataSetType::training) dropout->setEnabled(true);
    else dropout->setEnabled(false);
    
    if(!dropout->unset() && msg->time > time){
        dropout->nextStep(msg->time);
        time = msg->time;
    }
    
    forwardMessagePool->returnMessage(msg);
} 

void AsyncNeuralNode::InputNode::onRecv(BackwardPropagationMessage* msg) {
    if(receivedDelta.cols() != msg->delta.size()) receivedDelta = Eigen::MatrixXf::Zero(weights.size(),msg->delta.size());
    
    if(backwardDiscardMsgCheck(msg)) return;
    
    backwardSeenCount++;
    int index = dstWeightIndex[msg->src];
    receivedDelta.row(index) = msg->delta;
    
    backwardMessagePool->returnMessage(msg);
}