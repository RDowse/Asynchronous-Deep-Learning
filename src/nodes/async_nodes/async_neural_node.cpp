
#include "nodes/async_nodes/async_neural_node.h"
#include "messages/forward_propagation_message.h"
#include "messages/backward_propagation_message.h"

MessagePool<ForwardPropagationMessage>* AsyncNeuralNode::forwardMessagePool 
        = MessagePool<ForwardPropagationMessage>::getInstance();
MessagePool<BackwardPropagationMessage>* AsyncNeuralNode::backwardMessagePool  
        = MessagePool<BackwardPropagationMessage>::getInstance();

bool AsyncNeuralNode::forwardDiscardMsgCheck(ForwardPropagationMessage* msg){
    if(DataSetType::training != dataSetType) return false;
    
    if(curr_forward_batch > msg->batchNum){
        discardedForwardMessageCount++;
        forwardMessagePool->returnMessage(msg);
        return true; // ignore message
    }

    if(!ready && forwardSeenCount > (int)((float)incomingForwardEdges.size()*context->forwardDropTolerance) ){
        ready = true;
    }

    return false;
};

bool AsyncNeuralNode::backwardDiscardMsgCheck(BackwardPropagationMessage* msg){
    if(curr_backward_batch > msg->batchNum){
        discardedBackwardMessageCount++;
        backwardMessagePool->returnMessage(msg);
        return true; // ignore message
    }

    if(!ready && backwardSeenCount > (int)((float)incomingBackwardEdges.size()*context->backwardDropTolerance) ){
        ready = true;
    }

    return false;
};