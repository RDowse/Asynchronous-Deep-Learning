
/* 
 * File:   kfold.h
 * Author: ryan
 *
 * Created on 21 May 2017, 21:44
 */

#ifndef KFOLD_H
#define KFOLD_H

// source: https://sureshamrita.wordpress.com/2011/08/24/c-implementation-of-k-fold-cross-validation/
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
    std::vector<int> whichFoldToGo;
};

template<class In>
Kfold<In>::Kfold(int _k, In _beg, In _end) :
        beg(_beg), end(_end), K(_k) {
    if (K <= 0)
        throw std::runtime_error("The supplied value of K is =... One cannot create ... no of folds");

    //create the vector of integers
    int foldNo = 0;
    for (In i = beg; i != end; i++) {
        whichFoldToGo.push_back(++foldNo);
        if (foldNo == K)
            foldNo = 0;
    }
    if (!K)
        throw std::runtime_error("With this value of k (="")Equal division of the data is not possible");
    std::random_shuffle(whichFoldToGo.begin(), whichFoldToGo.end());
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

#endif /* KFOLD_H */

