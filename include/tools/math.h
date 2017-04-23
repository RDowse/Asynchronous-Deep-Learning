/* 
 * File:   math.h
 * Author: ryan
 *
 * Created on 04 April 2017, 20:51
 */

#ifndef MATH_H
#define MATH_H

#include <vector>
#include <map>
#include <cmath>
#include <ctime>
#include <random>
#include <algorithm>

namespace math{
    
    inline float activationSig(float x){
        return 1/(1+exp(-x));
    }
    
    inline float activationTan(float x){
        return 2 / (1 + exp(-2 * x)) - 1;
    }
    
    inline float activationStep(float x){
        return x > 0 ? 1 : -1;
    }
    
    static float randomFloat(float min, float max) {
        return (max - min) * ((((float) rand()) / (float) RAND_MAX)) + min;
    }

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
    
    // source: https://sureshamrita.wordpress.com/2011/08/24/c-implementation-of-k-fold-cross-validation/
    // TODO: replace with own version
    template<class In>
    class Kfold {
    public:
        Kfold(int k, In _beg, In _end);
        template<class Out>
        void getFold(int foldNo, Out training, Out testing);
        template<class Out>
        void getFold(int foldNo, Out training, Out testing, Out trainingLabel, Out testingLabel);
    private:
        In beg;
        In end;
        int K; //how many folds in this
        vector<int> whichFoldToGo;
    };

    template<class In>
    Kfold<In>::Kfold(int _k, In _beg, In _end) :
            beg(_beg), end(_end), K(_k) {
        if (K <= 0)
            throw runtime_error("The supplied value of K is =... One cannot create ... no of folds");

        //create the vector of integers
        int foldNo = 0;
        for (In i = beg; i != end; i++) {
            whichFoldToGo.push_back(++foldNo);
            if (foldNo == K)
                foldNo = 0;
        }
        if (!K)
            throw runtime_error("With this value of k (="")Equal division of the data is not possible");
        random_shuffle(whichFoldToGo.begin(), whichFoldToGo.end());
    }

    template<class In>
    template<class Out>
    void Kfold<In>::getFold(int foldNo, Out training, Out testing) {

        int k = 0;
        In i = beg;
        while (i != end) {
            if (whichFoldToGo[k++] == foldNo) {
                *testing++ = *i++;
            } else
                *training++ = *i++;
        }
    }
}

#endif /* MATH_H */

