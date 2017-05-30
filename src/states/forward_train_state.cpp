
#include "states/forward_train_state.h"
#include "nodes/neural_node.h"
#include "nodes/pardata_nodes/parallel_data_neural_node.h"

template<> void ForwardTrainState<NeuralNode>::onSend(NeuralNode* n, vector<Message*>& msgs){
    n->sendForwardMsgs(msgs);
}

template<> bool ForwardTrainState<NeuralNode>::readyToSend(NeuralNode* n){
    return n->readyToSendForward();
}

template<> void ForwardTrainState<ParallelDataNeuralNode>::onSend(ParallelDataNeuralNode* n, vector<Message*>& msgs){
    n->sendForwardMsgs(msgs);
}

template<> bool ForwardTrainState<ParallelDataNeuralNode>::readyToSend(ParallelDataNeuralNode* n){
    return n->readyToSendForward();
}

