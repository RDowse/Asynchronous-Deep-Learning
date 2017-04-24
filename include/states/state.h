/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   state.h
 * Author: ryan
 *
 * Created on 24 April 2017, 00:14
 */

#ifndef STATE_H
#define STATE_H

#include <iostream>
#include <memory>
#include <vector>

class NeuralNode;
class Node;
class Message;

using namespace std;

class State{
public:
    State(){}
    virtual void onSend(NeuralNode* n, vector<shared_ptr<Message>>& msgs);
    virtual void onSend(Node* n, vector<shared_ptr<Message>>& msgs);
    
    virtual bool readyToSend(NeuralNode* n);
    virtual bool readyToSend(Node* n);
    
    // Default call if method not overridden
    void notImplemented(){ std::cout << "State onSend not implemented for this type";  }
};

#endif /* STATE_H */

