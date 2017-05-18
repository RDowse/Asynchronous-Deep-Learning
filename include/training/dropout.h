
/* 
 * File:   dropout.h
 * Author: ryan
 *
 * Created on 16 May 2017, 17:54
 */

#ifndef DROPOUT_H
#define DROPOUT_H

#include <vector>
#include <cassert>

using namespace std;

class Dropout{
public:
    Dropout(){}
    void getActiveNodes(vector<int>& activeIndex, int nNodes, int bias, int step){
        assert(isPrime(nNodes));
        assert(bias < nNodes && bias >= 0);
        assert(step < nNodes && step >= 1);
        int acc = 0;
        int n = ceil(float(nNodes)/2);
        activeIndex = vector<int>(n);
        for(int i = 0; i < n; ++i){
            activeIndex[i] = perm(bias, acc, nNodes) % nNodes;
            acc+=step;
        }
    }
private:
    int perm(int bias, int acc, int prime){
        return bias + (acc % prime);
    }
    // source: http://www.cplusplus.com/forum/general/1125/#msg3850
    bool isPrime (int num)
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
};

#endif /* DROPOUT_H */

