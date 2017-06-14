
#include "nodes/async_nodes/async_neural_node.h"
#include "messages/forward_propagation_message.h"
#include "messages/backward_propagation_message.h"

MessagePool<ForwardPropagationMessage>* AsyncNeuralNode::forwardMessagePool 
        = MessagePool<ForwardPropagationMessage>::getInstance();
MessagePool<BackwardPropagationMessage>* AsyncNeuralNode::backwardMessagePool  
        = MessagePool<BackwardPropagationMessage>::getInstance();

bool AsyncNeuralNode::forwardDiscardMsgCheck(ForwardPropagationMessage* msg){
    if(curr_forward_batch > msg->time){
        discardedForwardMessageCount++;
        forwardMessagePool->returnMessage(msg);
        return true; // ignore message
    }

    if(!receivedFirstMessage && forwardSeenCount > (int)((float)incomingForwardEdges.size()*context->forwardDropTolerance) ){
        receivedFirstMessage = true;
        forwardTime = context->stepTime;
    }

    return false;
};

bool AsyncNeuralNode::backwardDiscardMsgCheck(BackwardPropagationMessage* msg){
    if(curr_backward_batch > msg->time){
        discardedBackwardMessageCount++;
        backwardMessagePool->returnMessage(msg);
        return true; // ignore message
    }

    if(!receivedFirstMessage && backwardSeenCount > (int)((float)incomingBackwardEdges.size()*context->backwardDropTolerance) ){
        receivedFirstMessage = true;
        backwardTime = context->stepTime;
    }

    return false;
};