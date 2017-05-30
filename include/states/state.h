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

class Node;
class Message;

using namespace std;

template<typename TNode>
class State{
public:
    State(){}
    
    virtual void onSend(TNode* n, vector<Message*>& msgs)=0;
    
    virtual bool readyToSend(TNode* n)=0;
};

#endif /* STATE_H */

