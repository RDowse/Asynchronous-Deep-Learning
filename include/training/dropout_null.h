
/* 
 * File:   dropout_null.h
 * Author: ryan
 *
 * Created on 23 May 2017, 22:08
 */

#ifndef DROPOUT_NULL_H
#define DROPOUT_NULL_H

#include "training/dropout_strategy.h"

class DropoutNull: public DropoutStrategy{
public:
    DropoutNull(){}
    bool unset(){return true;};
    void nextStep(int currTime){}
    bool isActive(){return true;}
    bool isPrevLayerNodeActive(int i){return true;}
    bool isNextLayerNodeActive(int i){return true;}
    
    bool readyToSendForward(int forwardSeenCount){return false;}
    bool readyToSendBackward(int backwardSeenCount){return false;}
};

#endif /* DROPOUT_NULL_H */

