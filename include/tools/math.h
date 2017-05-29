/* 
 * File:   math.h
 * Author: ryan
 *
 * Created on 04 April 2017, 20:51
 */

#ifndef MATH_H
#define MATH_H

#include <Eigen/Dense>

#include <vector>
#include <map>
#include <cmath>
#include <ctime>
#include <random>
#include <algorithm>
#include <cassert>
#include <iostream>

namespace math{
    
    /*
     * Activation functions
     */
    inline float activationSig(float x){
        return 1 / (1 + exp(-x));
    }
    
    inline float activationTanH(float x){
        return 2 / (1 + exp(-2 * x)) - 1;
    }
    
    inline float activationStep(float x){
        return x > 0 ? 1 : -1;
    }
    
    inline float deltaActivationSig(float x){
        return x*(1-x);
    }
        
    inline float deltaActivationTanH(float x){
        return 1 - pow(x,2);
    }

    inline Eigen::MatrixXf blockDeltaActivationSig(Eigen::MatrixXf x){
        auto ones = Eigen::MatrixXf::Ones(x.rows(),x.cols());
        return x.transpose()*(ones - x);
    }

    /*
     *  Dropout related functions
     */
    inline int permutation(int bias, int acc, int prime){
        return (bias + acc) % prime;
    }
        
    // source: http://www.cplusplus.com/forum/general/1125/#msg3850
    inline bool isPrime (int num)
    {
        if (num <=1)
            return false;
        else if (num == 2)         
            return true;
        else if (num % 2 == 0)
            return false;
        else
        {
            bool prime = true;
            int divisor = 3;
            double num_d = static_cast<double>(num);
            int upperLimit = static_cast<int>(sqrt(num_d) +1);

            while (divisor <= upperLimit)
            {
                if (num % divisor == 0)
                    prime = false;
                divisor +=2;
            }
            return prime;
        }
    }
}

#endif /* MATH_H */

