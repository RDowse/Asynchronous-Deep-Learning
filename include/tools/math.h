/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   math.h
 * Author: ryan
 *
 * Created on 04 April 2017, 20:51
 */

#ifndef MATH_H
#define MATH_H

#include <vector>

namespace math{
    template<typename T>
    void removeConstantCols(const std::vector<std::vector<T>>& X, std::vector<int>& removedIndex){
        for(int j = 0; j < X[0].size(); ++j){
            bool remove = true;
            auto tmp = X[0][j];
            for(int i = 0; i < X.size(); ++i){
                if(tmp!=X[i][j]){
                    remove = false;
                    break;
                }
            }
            if(remove) removedIndex.push_back(j);
        }
    }
}

#endif /* MATH_H */

