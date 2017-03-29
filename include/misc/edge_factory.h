/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   EdgeFactory.h
 * Author: ryan
 *
 * Created on 13 February 2017, 20:16
 */

#ifndef EDGEFACTORY_H
#define EDGEFACTORY_H

#include "misc/edge.h"
#include "tools/logging.h"

#include <map>
#include <memory>
#include <string>
#include <iostream>

using namespace std;

class Edge;

class EdgeFactory{    
public:
    typedef shared_ptr<Edge>(*construction_func)(shared_ptr<Node> dst, shared_ptr<Node> src, unsigned delay);
    typedef std::map<string,construction_func> map_type;
    
    // create an instance of the object name
    static shared_ptr<Edge> createInstance(string const& type, shared_ptr<Node> dst, shared_ptr<Node> src, unsigned delay) { // add some struct for parameter settings
        map_type::iterator it = getMap()->find(type);
        if (it == getMap()->end()){
            fprintf(stderr, "Error: %s is not registered.\n",type.c_str());
            return 0; // not registered
        }
        printf("Creating instance: %s\n",type.c_str());
        return it->second(dst,src,delay);
    }
protected:
    // stores currently registered node type constructors
    static map_type * m_map;
    
    static map_type * getMap() {
        if (!m_map) {
            m_map = new map_type();
        }
        return m_map; 
    }
};

template<typename T>
class EdgeRegister: public EdgeFactory{
public:
    EdgeRegister(const string& s) {
        printf("Registering: %s\n", s.c_str());
        getMap()->insert(std::pair<string,construction_func>(s, &edge::createT<T>));
    }
};
#endif /* EDGEFACTORY_H */
