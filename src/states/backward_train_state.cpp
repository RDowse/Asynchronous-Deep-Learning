
#include "states/backward_train_state.h"
#include "nodes/neural_node.h"
//#include "nodes/block_nodes/block_neural_node.h"

void BackwardTrainState::onSend(NeuralNode* n, vector<Message*>& msgs){
    n->sendBackwardMsgs(msgs);
}

bool BackwardTrainState::readyToSend(NeuralNode* n){
    return n->readyToSendBackward();
}


//void BackwardTrainState::onSend(BlockNeuralNode* n, vector<Message*>& msgs){
//    n->sendBackwardMsgs(msgs);
//}
//
//bool BackwardTrainState::readyToSend(BlockNeuralNode* n){
//    return n->readyToSendBackward();
//}