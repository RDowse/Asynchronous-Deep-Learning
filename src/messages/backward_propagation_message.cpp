
#include "messages/backward_propagation_message.h"
#include "nodes/node.h"

bool BackwardPropagationMessage::dispatchTo(shared_ptr<Node> handler){
    handler->onRecv(shared_from_this());
}

bool BackwardPropagationMessage::dispatchFrom(shared_ptr<Node> handler){
    handler->onSend(shared_from_this());
}