
#include "states/predict_state.h"
#include "nodes/neural_node.h"

void PredictState::onSend(NeuralNode* n, vector<Message*>& msgs){
    n->sendForwardMsgs(msgs);
}

bool PredictState::readyToSend(NeuralNode* n){
    return n->readyToSendForward();
}