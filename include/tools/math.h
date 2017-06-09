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
        
    inline float deltaActivationSig(float x){
        return x*(1-x);
    }
    
    inline float activationTanH(float x){
        return 2 / (1 + exp(-2 * x)) - 1;
    }
        
    inline float deltaActivationTanH(float x){
        return 1 - pow(x,2);
    }

    inline float activationLinear(float x){
        if(abs(x) < 2 ) return 0.5*x;
        return x >= 2 ? 1 : -1; 
    }
    
    inline float deltaActivationLinear(float x){
        return 0.5;
    }
    /*
     * Cost functions
     */
    // single sample
    inline float mse(Eigen::VectorXf target, Eigen::VectorXf output){
        Eigen::VectorXf diff = target - output;
        float tmp = diff.transpose()*diff;
        return 0.5*tmp;
    }
    
    inline float mse(Eigen::MatrixXf target, Eigen::MatrixXf output){
        int batchSize = target.rows();
        Eigen::MatrixXf diff = target - output;
        Eigen::MatrixXf tmp = diff.unaryExpr([](float d) {return 0.5*std::pow(d, 2);});
        Eigen::VectorXf sum_error(batchSize);
        for(int i = 0; i < batchSize; ++i)
            sum_error(i) = tmp.row(i).sum();
        return sum_error.sum();
    }
    
    // BlockNeuralNode implementation
    inline Eigen::MatrixXf blockDeltaActivationSig(Eigen::MatrixXf x){
        auto ones = Eigen::MatrixXf::Ones(x.rows(),x.cols());
        return x.transpose()*(ones - x);
    }
    
    // Verification tool
    inline float gradCheck(float (*func)(float,float), Eigen::VectorXf weights, int numChecks){
        float delta = 0.001;
        float sum_error = 0;
        
//        cout << "Performing gradient checking" << endl;
//        for(int i = 0; i < numChecks; ++i){
//            auto T0 = weights;
//            auto T1 = weights;
//            int j = rand()%weights.size();
//            T0(j) = T0(j) - delta;
//            T1(j) = T1(j) + delta;
//            
//            float f0,f1; // cost func
//            float g; // derivative
//            g_est = (f1-f0)/(2*delta);
//            error = abs()
//        }
        
        return sum_error/numChecks;
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

