
/* 
 * File:   dropout_strategy.h
 * Author: ryan
 *
 * Created on 23 May 2017, 22:00
 */

#ifndef DROPOUT_STRATEGY_H
#define DROPOUT_STRATEGY_H

class DropoutStrategy{
protected:
    bool enabled;
public:
    enum NodeType{input,hidden,output,bias};
    DropoutStrategy(bool _enabled): enabled(_enabled){}
    virtual bool unset(){return false;};
    virtual void nextStep(int currTime)=0;
    virtual bool isActive()=0;
    virtual bool isPrevLayerNodeActive(int i)=0;
    virtual bool isNextLayerNodeActive(int i)=0;
    
    virtual bool readyToSendForward(int forwardSeenCount)=0;
    virtual bool readyToSendBackward(int backwardSeenCount)=0;
    virtual void print(){}
    void setEnabled(bool _enabled){enabled = _enabled;}
    bool isEnabled(){return enabled;}
};

#endif /* DROPOUT_STRATEGY_H */

