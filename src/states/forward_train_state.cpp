
#include "states/forward_train_state.h"

#include "nodes/neural_node.h"
#include "nodes/async_nodes/async_neural_node.h"
#include "nodes/pardata_nodes/parallel_data_neural_node.h"

// NeuralNode
template<> void ForwardTrainState<NeuralNode>::onSend(NeuralNode* n, vector<Message*>& msgs){
    n->sendForwardMsgs(msgs);
}

template<> void ForwardTrainState<NeuralNode>::onSend(NeuralNode* n, vector<Message*>& msgs, int stateIndex){
    assert(0);
}

template<> bool ForwardTrainState<NeuralNode>::readyToSend(NeuralNode* n){
    return n->readyToSendForward();
}

template<> bool ForwardTrainState<NeuralNode>::readyToSend(NeuralNode* n, int stateIndex){
    assert(0);
}

// ParallelDataNeuralNode
template<> void ForwardTrainState<ParallelDataNeuralNode>::onSend(ParallelDataNeuralNode* n, vector<Message*>& msgs){
    assert(0);
}

template<> void ForwardTrainState<ParallelDataNeuralNode>::onSend(ParallelDataNeuralNode* n, vector<Message*>& msgs, int stateIndex){
    n->sendForwardMsgs(msgs,stateIndex);
}

template<> bool ForwardTrainState<ParallelDataNeuralNode>::readyToSend(ParallelDataNeuralNode* n){
    assert(0);
}

template<> bool ForwardTrainState<ParallelDataNeuralNode>::readyToSend(ParallelDataNeuralNode* n, int stateIndex){
    return n->readyToSendForward(stateIndex);
}

// Async nodes
template<> void ForwardTrainState<AsyncNeuralNode>::onSend(AsyncNeuralNode* n, vector<Message*>& msgs){
    n->sendForwardMsgs(msgs);
}

template<> void ForwardTrainState<AsyncNeuralNode>::onSend(AsyncNeuralNode* n, vector<Message*>& msgs, int stateIndex){
    assert(0);
}

template<> bool ForwardTrainState<AsyncNeuralNode>::readyToSend(AsyncNeuralNode* n){
    return n->readyToSendForward();
}

template<> bool ForwardTrainState<AsyncNeuralNode>::readyToSend(AsyncNeuralNode* n, int stateIndex){
    assert(0);
}