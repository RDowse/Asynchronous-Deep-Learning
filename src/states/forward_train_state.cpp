
#include "states/forward_train_state.h"
#include "nodes/neural_node.h"

void ForwardTrainState::onSend(NeuralNode* n, vector<Message*>& msgs){
    n->sendForwardMsgs(msgs);
}

bool ForwardTrainState::readyToSend(NeuralNode* n){
    return n->readyToSendForward();
}