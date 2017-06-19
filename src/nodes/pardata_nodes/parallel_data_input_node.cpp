
#include "nodes/pardata_nodes/parallel_data_input_node.h"
#include "messages/forward_propagation_message.h"
#include "messages/backward_propagation_message.h"
#include "tools/math.h"

std::string ParallelDataNeuralNode::InputNode::m_type = "Input";

void ParallelDataNeuralNode::InputNode::addEdge(Edge* e) {
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

bool ParallelDataNeuralNode::InputNode::sendForwardMsgs(vector<Message*>& msgs, int stateIndex){
    assert(readyToSendForward(stateIndex));
    
    if(!weights.size()) initWeights();
    Eigen::MatrixXf mat = activation[stateIndex]*weights.transpose();
    
    msgs.reserve(outgoingForwardEdges.size());
    assert(weights.size() == outgoingForwardEdges.size());
    for(unsigned i = 0; i < outgoingForwardEdges.size(); i++){
        if(dropout->isNextLayerNodeActive(i)){
            assert( 0 == outgoingForwardEdges[i]->msgStatus );
            auto msg = forwardMessagePool->getMessage();
            msg->src = id;
            msg->dst = outgoingForwardEdges[i]->dst->getId();
            msg->batchNum = batchNum;
            msg->dataSetType = dataSetType;
            msg->batchIndex = stateIndex;
            msg->updateNumber = updateCount;

            msg->activation = mat.col(i);
            msgs.push_back(msg);
        }
    }
    
    // reset 
    forwardSeenCount[stateIndex] = 0;
    delete state[stateIndex];
    state[stateIndex] = new BackwardTrainState<ParallelDataNeuralNode>();
}

bool ParallelDataNeuralNode::InputNode::sendBackwardMsgs(vector<Message*>& msgs, int stateIndex){
    assert(readyToSendBackward(stateIndex));
            
    // perform weight update first
    int batchSize = receivedDelta.cols();
    deltaWeights = (context->lr/context->numModels)*(receivedDelta * activation[stateIndex])/batchSize + context->alpha*deltaWeights;
    
    weights -= deltaWeights; // async update step  
    updateCount++; // track current number updates
    
    msgs.reserve(outgoingBackwardEdges.size());
    for(unsigned i = 0; i < outgoingBackwardEdges.size(); i++){
        assert( 0 == outgoingBackwardEdges[i]->msgStatus );
        auto msg = backwardMessagePool->getMessage();
        msg->src = id;
        msg->dst = outgoingBackwardEdges[i]->dst->getId();
        msg->batchNum = batchNum;
        msg->batchIndex = stateIndex;
        msgs.push_back(msg);
    }
    
    // reset
    backwardSeenCount[stateIndex] = 0;
    delete state[stateIndex];
    state[stateIndex] = new ForwardTrainState<ParallelDataNeuralNode>();
}

void ParallelDataNeuralNode::InputNode::onRecv(ForwardPropagationMessage* msg){
    activation[msg->batchIndex] = msg->activation;
    forwardSeenCount[msg->batchIndex]++;
    
    dataSetType = msg->dataSetType;
    
    forwardMessagePool->returnMessage(msg);
} 

void ParallelDataNeuralNode::InputNode::onRecv(BackwardPropagationMessage* msg) {
    if(receivedDelta.cols() != msg->delta.size()) receivedDelta = Eigen::MatrixXf::Zero(weights.size(),msg->delta.size());
    int index = dstWeightIndex[msg->src];
    receivedDelta.row(index) = msg->delta;
    backwardSeenCount[msg->batchIndex]++;
    
    backwardMessagePool->returnMessage(msg);
}