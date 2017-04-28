/* 
 * File:   message.h
 * Author: ryan
 *
 * Created on 19 March 2017, 16:53
 */

#ifndef MESSAGE_H
#define MESSAGE_H

#include <memory>

class Node;

using namespace std;

// Abstract message
class Message{
public:
    int src = 0;
    int dst = 0;
    // send message to node
    virtual bool dispatchTo(Node* handler)=0;
};

#endif /* MESSAGE_H */