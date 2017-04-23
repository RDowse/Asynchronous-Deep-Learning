
#include "messages/backward_propagation_message.h"
#include "nodes/node.h"

bool BackwardPropagationMessage::dispatchTo(Node* handler){
    handler->onRecv(shared_from_this());
}