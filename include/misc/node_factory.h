/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   NodeFactory.h
 * Author: ryan
 *
 * Created on 13 February 2017, 20:16
 */

#ifndef NODEFACTORY_H
#define NODEFACTORY_H

#include "nodes/node.h"

#include <map>
#include <string>
#include <iostream>

using namespace std;

class NodeFactory{    
public:
    typedef std::map<string, Node*(*)(shared_ptr<Graph>) > map_type;

    static Node * createInstance(string const& type, shared_ptr<Graph> graph) { // add some struct for parameter settings
        map_type::iterator it = getMap()->find(type);
        if (it == getMap()->end()){
            fprintf(stderr, "Error: %s is not registered.\n",type.c_str());
            return 0; // not registered
        }
        printf("Creating instance: %s\n",type.c_str());
        return it->second(graph);
    }
protected:
    static map_type * m_map;
    
    static map_type * getMap() {
        if (!m_map) {
            m_map = new map_type();
        }
        return m_map; 
    }
};

template<typename T>
class NodeRegister: public NodeFactory{
public:
    NodeRegister(const string& s) {
        printf("Registering: %s\n", s.c_str());
        getMap()->insert(std::pair<string,Node*(*)(shared_ptr<Graph>)>(s, &createT<T>));
    }
};
#endif /* NODEFACTORY_H */

