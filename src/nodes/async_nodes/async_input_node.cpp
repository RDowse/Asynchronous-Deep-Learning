
#include "nodes/async_nodes/async_input_node.h"
#include "messages/forward_propagation_message.h"
#include "messages/backward_propagation_message.h"
#include "tools/math.h"

std::string AsyncNeuralNode::InputNode::m_type = "Input";

void AsyncNeuralNode::InputNode::addEdge(Edge* e) {
     // add to original edge sets
     Node::addEdge(e);
     // check edge belongs to this node
     if(e->src->getId() == id){
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
     } else if(e->dst->getId() == id){
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
            msg->src = id;
            msg->dst = outgoingForwardEdges[i]->dst->getId();
            msg->batchNum = curr_forward_batch;
            msg->dataSetType = dataSetType;

            if(context->epoch==context->maxEpoch && dataSetType == DataSetType::testing) context->insertHist(mat.col(i));
            
            msg->activation = mat.col(i);
            msgs.push_back(msg);
            
            numMessagesSentForward++;
        }
    }
    
    if(DataSetType::training == dataSetType) curr_forward_batch++;
    
    forwardSeenCount = 0;
    
    // reset 
    if(dataSetType == DataSetType::training) swapState<BackwardTrainState<AsyncNeuralNode>>();
}

bool AsyncNeuralNode::InputNode::sendBackwardMsgs(vector<Message*>& msgs){
    
    // perform weight update first
    int batchSize = receivedDelta.cols();
    deltaWeights = context->lr*(receivedDelta * activation)/batchSize + context->alpha*deltaWeights;
    
    weights -= deltaWeights;
    weights = context->regularizationFnc(weights, context->c);
    
    msgs.reserve(outgoingBackwardEdges.size());
    for(unsigned i = 0; i < outgoingBackwardEdges.size(); i++){
        assert( 0 == outgoingBackwardEdges[i]->msgStatus );
        auto msg = backwardMessagePool->getMessage();
        msg->src = id;
        msg->dst = outgoingBackwardEdges[i]->dst->getId();
        msg->batchNum = curr_backward_batch;

        msgs.push_back(msg);
        
        numMessagesSentBackward++;
    }
    
    curr_backward_batch++;
    ready = false;
    
    // reset delta values
    receivedDelta.Zero(receivedDelta.rows(),receivedDelta.cols());
    
    backwardSeenCount = 0;
    swapState<ForwardTrainState<AsyncNeuralNode>>();
}

void AsyncNeuralNode::InputNode::onRecv(ForwardPropagationMessage* msg){
    activation = msg->activation;
    
    dataSetType = msg->dataSetType;
    
    if(dataSetType==DataSetType::training) dropout->setEnabled(true);
    else dropout->setEnabled(false);
    
    if(!dropout->unset() && msg->batchNum > batchNum && dataSetType==DataSetType::training){
        dropout->nextStep(msg->batchNum);
        batchNum = msg->batchNum;
        curr_forward_batch = msg->batchNum;
        curr_backward_batch = msg->batchNum;
    }
    
    forwardSeenCount++;
    forwardMessagePool->returnMessage(msg);
} 

void AsyncNeuralNode::InputNode::onRecv(BackwardPropagationMessage* msg) {
    if(receivedDelta.cols() != msg->delta.size()) receivedDelta = Eigen::MatrixXf::Zero(weights.size(),msg->delta.size());
    
    if(backwardDiscardMsgCheck(msg)) return;
    
    int index = dstWeightIndex[msg->src];
    receivedDelta.row(index) = msg->delta;
    
    backwardSeenCount++;
    backwardMessagePool->returnMessage(msg);
}