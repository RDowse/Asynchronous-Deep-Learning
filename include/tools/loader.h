/* 
 * File:   loader.h
 * Author: ryan
 *
 * Created on 28 March 2017, 21:57
 */

#ifndef LOADER_H
#define LOADER_H

#include "simulator.h"
#include "misc/node_factory.h"
#include "graphs/graph_settings.h"
#include "tools/logging.h"

#include "nodes/node.h"
#include "nodes/input_node.h"
#include "nodes/hidden_node.h"
#include "nodes/output_node.h"
#include "nodes/bias_node.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <memory>
#include <random>

using namespace std;

class Loader{
public:    
    static string readType(int& lineNumber, ifstream& src){
        if(src.is_open()){
            string type;
            std::stringstream err;
            expect(lineNumber,src,"POETSGraph");
            if(!(stringstream(nextline(lineNumber,src))>>type)){
                err<<"At line "<<lineNumber<<" : Couldn't read type";
                throw std::runtime_error(err.str());              
            }
            return type;
        } else {
            printf("No file open \n");
        }
        return NULL;
    }
    
    static void readHeader(int& lineNumber, ifstream& src, shared_ptr<GraphSettings> settings,
                            int& nNodes, int& nEdges){
        if(src.is_open()){
            std::stringstream err;
            expect(lineNumber,src,"BeginHeader");
            // Parameter count
            int paramCount;
            if(!(stringstream(nextline(lineNumber,src))>>paramCount)){
                err<<"At line "<<lineNumber<<" : Couldn't read parameter count";
                throw std::runtime_error(err.str());              
            }

            // Parameters, package depending on graph type
            vector<int> params;
            stringstream ss(nextline(lineNumber,src));
            for(int i = 0; i < paramCount; ++i){
                int tmpParam;
                ss >> tmpParam;
                params.push_back(tmpParam);
            }
            settings->setParameters(params);

            // Node and edge count
            if(!(stringstream(nextline(lineNumber,src))>>nNodes>>nEdges)){
                err<<"At line "<<lineNumber<<" : Couldn't read nNodes, nEdges";
                throw std::runtime_error(err.str());              
            }

            expect(lineNumber,src,"EndHeader");
        } else {
            printf("No file open\n");
        }
    } 
    
    template<typename TNode>
    static void readBody(int& lineNumber, std::ifstream &src, shared_ptr<GraphSettings> settings, Simulator<TNode>& sim,
                            int nNodes, int nEdges){
        if(src.is_open()){
            std::stringstream err;
            sim.setGraphSettings(settings);
            
            expect(lineNumber,src,"BeginNodes");
            for(int i = 0; i < nNodes; ++i){
                int id;
                string type;
                if(!(stringstream(nextline(lineNumber,src))>>type>>id)){
                    err<<"At line "<<lineNumber<<" : Couldn't read node";
                    throw std::runtime_error(err.str());              
                }   
                sim.addNode(NodeFactory::createInstance(type,settings));
            }
            expect(lineNumber,src,"EndNodes");

            expect(lineNumber,src,"BeginEdges");
            for(int i = 0; i < nEdges; ++i){
                int srcIndex, dstIndex, delay;
                if(!(stringstream(nextline(lineNumber,src))>>srcIndex>>dstIndex>>delay)){
                    err<<"At line "<<lineNumber<<" : Couldn't read edge";
                    throw std::runtime_error(err.str());              
                }
                sim.addEdge(srcIndex,dstIndex,delay);
            }
            expect(lineNumber,src,"EndEdges");
        } else {
            printf("No file open\n");
        }
    }
    
private:    
    static string nextline(int& lineNumber, std::ifstream &src){
        lineNumber++;
        string line;
        if(!std::getline(src,line)){
            std::stringstream err;
            err<<"Couldn't read line number"<<lineNumber<<".\n";
            throw std::runtime_error(err.str());
        }
        return line;
    }
    
    static void expect(int& lineNumber, std::ifstream &src, const string& s){
        string token=nextline(lineNumber,src);
        if(token!=s){
            std::stringstream err;
            err<<"At line "<<lineNumber<<" : expecting '"<<s<<"', but got '"<<token<<"'";
            throw std::runtime_error(err.str());
        }
    }
};

#endif /* LOADER_H */