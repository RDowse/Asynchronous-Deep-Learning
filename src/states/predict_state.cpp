
#include "states/predict_state.h"

#include "nodes/neural_node.h"
#include "nodes/async_nodes/async_neural_node.h"
#include "nodes/pardata_nodes/parallel_data_neural_node.h"

// NeuralNode
template<> void PredictState<NeuralNode>::onSend(NeuralNode* n, vector<Message*>& msgs){
    n->sendForwardMsgs(msgs);
}

template<> void PredictState<NeuralNode>::onSend(NeuralNode* n, vector<Message*>& msgs, int stateIndex){
    assert(0);
}

template<> bool PredictState<NeuralNode>::readyToSend(NeuralNode* n){
    return n->readyToSendForward();
}

template<> bool PredictState<NeuralNode>::readyToSend(NeuralNode* n, int stateIndex){
    assert(0);
}


// ParallelDataNeuralNode
template<> void PredictState<ParallelDataNeuralNode>::onSend(ParallelDataNeuralNode* n, vector<Message*>& msgs){
    assert(0);
}

template<> void PredictState<ParallelDataNeuralNode>::onSend(ParallelDataNeuralNode* n, vector<Message*>& msgs, int stateIndex){
    n->sendForwardMsgs(msgs,stateIndex);
}

template<> bool PredictState<ParallelDataNeuralNode>::readyToSend(ParallelDataNeuralNode* n){
    assert(0);
}

template<> bool PredictState<ParallelDataNeuralNode>::readyToSend(ParallelDataNeuralNode* n, int stateIndex){
    return n->readyToSendForward(stateIndex);
}

// AsyncNode
template<> void PredictState<AsyncNeuralNode>::onSend(AsyncNeuralNode* n, vector<Message*>& msgs){
    n->sendForwardMsgs(msgs);
}

template<> void PredictState<AsyncNeuralNode>::onSend(AsyncNeuralNode* n, vector<Message*>& msgs, int stateIndex){
    assert(0);
}

template<> bool PredictState<AsyncNeuralNode>::readyToSend(AsyncNeuralNode* n){
    return n->readyToSendForward();
}

template<> bool PredictState<AsyncNeuralNode>::readyToSend(AsyncNeuralNode* n, int stateIndex){
    assert(0);
}