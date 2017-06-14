
#include "nodes/input_node.h"
#include "messages/forward_propagation_message.h"
#include "messages/backward_propagation_message.h"
#include "tools/math.h"

std::string NeuralNode::InputNode::m_type = "Input";

void NeuralNode::InputNode::addEdge(Edge* e) {
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

bool NeuralNode::InputNode::sendForwardMsgs(vector<Message*>& msgs){
    assert(readyToSendForward());
    
    if(!weights.size()) initWeights();
    
    Eigen::MatrixXf mat = activation*weights.transpose();
    
    msgs.reserve(outgoingForwardEdges.size());
    assert(weights.size() == outgoingForwardEdges.size());
    for(unsigned i = 0; i < outgoingForwardEdges.size(); i++){
        if(dropout->isNextLayerNodeActive(i) || dataSetType != DataSetType::training){
            assert( 0 == outgoingForwardEdges[i]->msgStatus );
            auto msg = forwardMessagePool->getMessage();
            msg->src = m_id;
            msg->dst = outgoingForwardEdges[i]->dst->getId();
            msg->batchNum = batchNum;
            msg->dataSetType = dataSetType;

            msg->activation = mat.col(i);
            msgs.push_back(msg);
        }
    }
    
    // reset 
    forwardSeenCount = 0;
    if(dataSetType==DataSetType::training) swapState<BackwardTrainState<NeuralNode>>();
}

bool NeuralNode::InputNode::sendBackwardMsgs(vector<Message*>& msgs){
    assert(readyToSendBackward());
            
    // perform weight update first
    int batchSize = receivedDelta.cols();
    deltaWeights = context->lr*(receivedDelta * activation)/batchSize + context->alpha*deltaWeights;

    weights -= deltaWeights; // update step  
    weights = context->regularizationFnc(weights, context->c);
    
    msgs.reserve(outgoingBackwardEdges.size());
    for(unsigned i = 0; i < outgoingBackwardEdges.size(); i++){
        assert( 0 == outgoingBackwardEdges[i]->msgStatus );
        auto msg = backwardMessagePool->getMessage();
        msg->src = m_id;
        msg->dst = outgoingBackwardEdges[i]->dst->getId();
        msg->batchNum = batchNum;

        msgs.push_back(msg);
    }
    
    // reset
    backwardSeenCount = 0;
    swapState<ForwardTrainState<NeuralNode>>();
}

void NeuralNode::InputNode::onRecv(ForwardPropagationMessage* msg){
    activation = msg->activation;
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

void NeuralNode::InputNode::onRecv(BackwardPropagationMessage* msg) {
    if(receivedDelta.cols() != msg->delta.size()) receivedDelta = Eigen::MatrixXf::Zero(weights.size(),msg->delta.size());
    int index = dstWeightIndex[msg->src];
    receivedDelta.row(index) = msg->delta;
    backwardSeenCount++;
    
    backwardMessagePool->returnMessage(msg);
}