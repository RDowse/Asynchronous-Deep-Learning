/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

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
#include "graphs/dnn_graph_settings.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <memory>

using namespace std;

class Loader{
public:
    static int load(string path, Simulator& sim){
        ifstream file;
        file.open(path);
        if(file.is_open()){
            string type;
            int lineNumber = 0, nNodes = 0, nEdges = 0;
            expect(lineNumber,file,"POETSGraph");
            type = readType(lineNumber,file);
            readHeader(lineNumber,file,nNodes,nEdges);
            readBody(lineNumber,file,sim,nNodes,nEdges);
        } else {
            printf("Unable to open file %s\n",path.c_str());
            return -1;
        }
        file.close();
    }
private:
    static string readType(int& lineNumber, ifstream& src){
        string type;
        std::stringstream err;
        if(!(stringstream(nextline(lineNumber,src))>>type)){
            err<<"At line "<<lineNumber<<" : Couldn't read type";
            throw std::runtime_error(err.str());              
        }
        return type;
    }
    
    static void readHeader(int& lineNumber,ifstream& src,
                            int& nNodes, int& nEdges){
        std::stringstream err;
        expect(lineNumber,src,"BeginHeader");
        // Parameter count
        int paramCount;
        if(!(stringstream(nextline(lineNumber,src))>>paramCount)){
            err<<"At line "<<lineNumber<<" : Couldn't read parameter count";
            throw std::runtime_error(err.str());              
        }
        
        // Parameters, package depending on graph type
        stringstream ss(nextline(lineNumber,src));
        for(int i = 0; i < paramCount; ++i){
            //todo
        }
        
        // Node and edge count
        if(!(stringstream(nextline(lineNumber,src))>>nNodes>>nEdges)){
            err<<"At line "<<lineNumber<<" : Couldn't read nNodes, nEdges";
            throw std::runtime_error(err.str());              
        }        

        expect(lineNumber,src,"EndHeader");    
    } 
    
    static void readBody(int& lineNumber, std::ifstream &src, Simulator& sim,
                            int nNodes, int nEdges){
        std::stringstream err;
        auto settings =  make_shared<DNNGraphSettings>();
        vector<shared_ptr<Node>> nodes;
        nodes.reserve(nNodes);
        
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
        
        vector<shared_ptr<Edge>> edges;
        nodes.reserve(nEdges);
        expect(lineNumber,src,"BeginEdges");
        for(int i = 0; i < nEdges; ++i){
            int srcIndex, dstIndex, delay;
            string type;
            if(!(stringstream(nextline(lineNumber,src))>>type>>srcIndex>>dstIndex>>delay)){
                err<<"At line "<<lineNumber<<" : Couldn't read edge";
                throw std::runtime_error(err.str());              
            }  
            sim.addEdge(type,srcIndex,dstIndex,delay);
        }
        expect(lineNumber,src,"EndEdges");
    }
    
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

