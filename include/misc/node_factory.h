/* 
 * File:   NodeFactory.h
 * Author: ryan
 *
 * Created on 13 February 2017, 20:16
 */

#ifndef NODEFACTORY_H
#define NODEFACTORY_H

#include "nodes/node.h"
#include "nodes/neural_node.h"
#include "nodes/async_nodes/async_neural_node.h"
#include "nodes/pardata_nodes/parallel_data_neural_node.h"

#include "tools/logging.h"

#include <map>
#include <memory>
#include <string>
#include <iostream>

using namespace std;

class NodeFactory{    
public:
    typedef Node* (*construction_func)(shared_ptr<GraphSettings>);
    typedef std::map<string,construction_func> map_type;
    
    // create an instance of the object name
    static Node* createInstance(string const& type, shared_ptr<GraphSettings> graphSettings) { // add some struct for parameter settings
        map_type::iterator it = getMap()->find(type);
        if (it == getMap()->end()){
            fprintf(stderr, "Error: %s is not registered.\n",type.c_str());
            return 0; // not registered
        }
        Logging::log(4, "Creating instance: %s\n", type.c_str());
        return it->second(graphSettings);
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
class NodeRegister: public NodeFactory{
public:
    NodeRegister(const string& s) {
        Logging::log(4, "Registering: %s", s.c_str());
        getMap()->insert(std::pair<string,construction_func>(s, &createT<T>));
    }
};

template<typename TNode>
void registerNodes(){
    typedef typename TNode::InputNode InputNode;
    typedef typename TNode::OutputNode OutputNode;
    typedef typename TNode::HiddenNode HiddenNode;
    typedef typename TNode::BiasNode BiasNode;
    typedef typename TNode::SyncNode SyncNode;
    
    NodeRegister<InputNode> m_reg_in(InputNode::m_type);
    NodeRegister<OutputNode> m_reg_out(OutputNode::m_type);
    NodeRegister<BiasNode> m_reg_bias(BiasNode::m_type);
    NodeRegister<HiddenNode> m_reg_hidden(HiddenNode::m_type);
    NodeRegister<SyncNode> m_reg_sync(SyncNode::m_type);
}

#endif /* NODEFACTORY_H */

