
#include "states/backward_train_state.h"
#include "nodes/neural_node.h"
#include "nodes/pardata_nodes/parallel_data_neural_node.h"

template<> void BackwardTrainState<NeuralNode>::onSend(NeuralNode* n, vector<Message*>& msgs){
    n->sendBackwardMsgs(msgs);
}

template<> bool BackwardTrainState<NeuralNode>::readyToSend(NeuralNode* n){
    return n->readyToSendBackward();
}

template<> void BackwardTrainState<ParallelDataNeuralNode>::onSend(ParallelDataNeuralNode* n, vector<Message*>& msgs){
    n->sendBackwardMsgs(msgs);
}

template<> bool BackwardTrainState<ParallelDataNeuralNode>::readyToSend(ParallelDataNeuralNode* n){
    return n->readyToSendBackward();
}