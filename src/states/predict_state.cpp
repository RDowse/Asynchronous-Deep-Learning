
#include "states/predict_state.h"
#include "nodes/neural_node.h"
#include "nodes/block_nodes/block_neural_node.h"

void PredictState::onSend(NeuralNode* n, vector<Message*>& msgs){
    n->sendForwardMsgs(msgs);
}

bool PredictState::readyToSend(NeuralNode* n){
    return n->readyToSendForward();
}


void PredictState::onSend(BlockNeuralNode* n, vector<Message*>& msgs){
    n->sendForwardMsgs(msgs);
}

bool PredictState::readyToSend(BlockNeuralNode* n){
    return n->readyToSendForward();
}