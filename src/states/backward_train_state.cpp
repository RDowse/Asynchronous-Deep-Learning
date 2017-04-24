
#include "states/backward_train_state.h"
#include "nodes/neural_node.h"

void BackwardTrainState::onSend(NeuralNode* n, vector<shared_ptr<Message>>& msgs){
    n->sendBackwardMsgs(msgs);
}

bool BackwardTrainState::readyToSend(NeuralNode* n){
    return n->readyToSendBackward();
}