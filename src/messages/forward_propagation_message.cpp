
#include "messages/forward_propagation_message.h"
#include "nodes/node.h"

bool ForwardPropagationMessage::dispatchTo(shared_ptr<Node> handler) {
    handler->onRecv(shared_from_this());
}