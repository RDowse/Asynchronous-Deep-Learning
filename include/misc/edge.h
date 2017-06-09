/* 
 * File:   edge.h
 * Author: ryan
 *
 * Created on 19 March 2017, 16:56
 */

#ifndef EDGE_H
#define EDGE_H

using namespace std;

class Message;
class Node;

class Edge{ 
public:
    unsigned delay;
    Message* msg;
    Node* dst;
    Node* src;
    unsigned msgStatus = 0; // empty -> ready -> inflight 
    
    Edge(Node* src, Node* dst, unsigned delay) :
    dst(dst), src(src), delay(delay), msgStatus(0) {
    }

    unsigned getDelay()const {
        return delay;
    }
};
#endif /* EDGE_H */