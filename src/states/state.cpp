
#include "states/state.h"
#include "nodes/neural_node.h"
//#include "nodes/block_nodes/block_neural_node.h"
#include "nodes/node.h"

void State::onSend(NeuralNode* n, vector<Message*>& msgs){ 
    notImplemented(); 
}

void State::onSend(Node* n, vector<Message*>& msgs){ 
    notImplemented(); 
}

//void State::onSend(BlockNeuralNode* n, vector<Message*>& msgs){ 
//    notImplemented(); 
//}

bool State::readyToSend(NeuralNode* n){
    notImplemented();
    return false;
}

//bool State::readyToSend(BlockNeuralNode* n){
//    notImplemented();
//    return false;
//}

bool State::readyToSend(Node* n){
    notImplemented();
    return false;
}