
/* 
 * File:   weight_initialisers.h
 * Author: ryan
 *
 * Created on 21 May 2017, 21:46
 */

#ifndef WEIGHT_INITIALISERS_H
#define WEIGHT_INITIALISERS_H

#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <random>
#include <cassert>

namespace weight_init{
    inline float randomFloat(float min, float max) {
        assert(min < max);
        return (max - min) * ((((float) rand()) / (float) RAND_MAX)) + min;
    } 

    // https://stats.stackexchange.com/a/186351 ref for formula
    inline void initWeights(std::vector<float>& weights, int nFanIn, int nFanOut){
        float r = sqrt(6.0/(float(nFanOut)+float(nFanIn)));
        for(int i = 0; i < weights.size(); ++i)
            weights[i] = randomFloat(-r, r);
    }

    inline void initWeights(Eigen::VectorXf& weights, int nFanIn, int nFanOut){
        float r = sqrt(6.0/(float(nFanOut)+float(nFanIn)));
        for(int i = 0; i < weights.size(); ++i)
            weights(i) = randomFloat(-r, r);
    }

    inline void initBlockWeights(Eigen::MatrixXf& weights, int nFanIn, int nFanOut){
        float r = sqrt(6.0/(float(nFanOut)+float(nFanIn)));
        for(int col = 0; col < weights.cols(); ++col){
            for(int row = 0; row < weights.rows(); ++row){
                weights(row,col) = randomFloat(-r, r);
            }
        }
    }
}

#endif /* WEIGHT_INITIALISERS_H */

