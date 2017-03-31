
#include "messages/forward_propagation_message.h"
#include "nodes/node.h"

bool ForwardPropagationMessage::dispatchTo(shared_ptr<Node> handler) {
    handler->onRecv(shared_from_this());
}
// prepare message sent from node
bool ForwardPropagationMessage::dispatchFrom(shared_ptr<Node> handler) {
    handler->onSend(shared_from_this());
}