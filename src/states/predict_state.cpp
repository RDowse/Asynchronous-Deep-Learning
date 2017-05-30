
#include "states/predict_state.h"
#include "nodes/neural_node.h"
#include "nodes/pardata_nodes/parallel_data_neural_node.h"

template<> void PredictState<NeuralNode>::onSend(NeuralNode* n, vector<Message*>& msgs){
    n->sendForwardMsgs(msgs);
}

template<> bool PredictState<NeuralNode>::readyToSend(NeuralNode* n){
    return n->readyToSendForward();
}

template<> void PredictState<ParallelDataNeuralNode>::onSend(ParallelDataNeuralNode* n, vector<Message*>& msgs){
    n->sendForwardMsgs(msgs);
}

template<> bool PredictState<ParallelDataNeuralNode>::readyToSend(ParallelDataNeuralNode* n){
    return n->readyToSendForward();
}
