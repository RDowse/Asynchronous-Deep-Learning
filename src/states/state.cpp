
#include "states/state.h"
#include "nodes/neural_node.h"
#include "nodes/node.h"

void State::onSend(NeuralNode* n, vector<shared_ptr<Message>>& msgs){ 
    notImplemented(); 
}

void State::onSend(Node* n, vector<shared_ptr<Message>>& msgs){ 
    notImplemented(); 
}

bool State::readyToSend(NeuralNode* n){
    notImplemented();
    return false;
}

bool State::readyToSend(Node* n){
    notImplemented();
    return false;
}