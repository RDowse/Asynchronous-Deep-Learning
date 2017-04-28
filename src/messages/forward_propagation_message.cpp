
#include "messages/forward_propagation_message.h"
#include "nodes/node.h"

bool ForwardPropagationMessage::dispatchTo(Node* handler) {
    handler->onRecv(this);
}