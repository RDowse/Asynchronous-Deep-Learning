/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   Data.h
 * Author: ryan
 *
 * Created on 01 April 2017, 01:37
 */

#ifndef DATA_H
#define DATA_H

#include <vector>

template<typename LABEL, typename DATA>
class Data{
public:
    enum Separator{comma,dot};
    std::vector<LABEL> label;
    std::vector<DATA> data;
};

#endif /* DATA_H */

