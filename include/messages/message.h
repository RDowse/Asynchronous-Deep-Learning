/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   message.h
 * Author: ryan
 *
 * Created on 19 March 2017, 16:53
 */

#ifndef MESSAGE_H
#define MESSAGE_H

#include <string>
#include <vector>

// Generic message
class Message{
public:
    std::string type;
    int src;    // Probably not needed. Msgs just passed onto the next node.
    int dst;
    std::vector<float> data; // set data size that must be unpacked
};

#endif /* MESSAGE_H */

