
#include "states/backward_train_state.h"

#include "nodes/neural_node.h"
#include "nodes/async_nodes/async_neural_node.h"
#include "nodes/pardata_nodes/parallel_data_neural_node.h"

// NeuralNode
template<> void BackwardTrainState<NeuralNode>::onSend(NeuralNode* n, vector<Message*>& msgs){
    n->sendBackwardMsgs(msgs);
}

template<> void BackwardTrainState<NeuralNode>::onSend(NeuralNode* n, vector<Message*>& msgs, int stateIndex){
    assert(0);
}

template<> bool BackwardTrainState<NeuralNode>::readyToSend(NeuralNode* n){
    return n->readyToSendBackward();
}

template<> bool BackwardTrainState<NeuralNode>::readyToSend(NeuralNode* n, int stateIndex){
    assert(0);
}

// ParallelDataNeuralNode
template<> void BackwardTrainState<ParallelDataNeuralNode>::onSend(ParallelDataNeuralNode* n, vector<Message*>& msgs){
    assert(0);
}

template<> void BackwardTrainState<ParallelDataNeuralNode>::onSend(ParallelDataNeuralNode* n, vector<Message*>& msgs, int stateIndex){
    n->sendBackwardMsgs(msgs, stateIndex);
}

template<> bool BackwardTrainState<ParallelDataNeuralNode>::readyToSend(ParallelDataNeuralNode* n){
    assert(0);
}

template<> bool BackwardTrainState<ParallelDataNeuralNode>::readyToSend(ParallelDataNeuralNode* n, int stateIndex){
    return n->readyToSendBackward(stateIndex);
}

// AsyncNeuralNode
template<> void BackwardTrainState<AsyncNeuralNode>::onSend(AsyncNeuralNode* n, vector<Message*>& msgs){
    n->sendBackwardMsgs(msgs);
}

template<> void BackwardTrainState<AsyncNeuralNode>::onSend(AsyncNeuralNode* n, vector<Message*>& msgs, int stateIndex){
    assert(0);
}
    
template<> bool BackwardTrainState<AsyncNeuralNode>::readyToSend(AsyncNeuralNode* n){
    return n->readyToSendBackward();
}

template<> bool BackwardTrainState<AsyncNeuralNode>::readyToSend(AsyncNeuralNode* n, int stateIndex){
    assert(0);
}