/* 
 * File:   state.h
 * Author: ryan
 *
 * Created on 24 April 2017, 00:14
 */

#ifndef STATE_H
#define STATE_H

#include <iostream>
#include <vector>

class BlockNeuralNode;
class NeuralNode;
class Node;
class Message;

using namespace std;

class State{
public:
    State(){}
    virtual void onSend(BlockNeuralNode* n, vector<Message*>& msgs);
    virtual void onSend(NeuralNode* n, vector<Message*>& msgs);
    virtual void onSend(Node* n, vector<Message*>& msgs);

    virtual bool readyToSend(BlockNeuralNode* n);    
    virtual bool readyToSend(NeuralNode* n);
    virtual bool readyToSend(Node* n);
    
    // Default call if method not overridden
    void notImplemented(){ std::cout << "State onSend not implemented for this type";  }
};

#endif /* STATE_H */

