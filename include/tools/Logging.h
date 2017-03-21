
#include <stdexcept>
#include <stdarg.h> 
#include <cstdio>

class Logging{
public: 
    static int m_logLevel;
    static void log(int level, const char *msg, ...)
    {
        if(level <= m_logLevel){
            char localBuffer[256];
            char *globalBuffer=0;
            char *buffer=localBuffer;
            
            
            va_list va;
            va_start(va, msg);
            int n=vsnprintf(buffer, sizeof(localBuffer), msg, va);
            va_end(va);
            
            if(n<=0){
                throw std::runtime_error("log failure.");
            }
            
            if(n >= (int)sizeof(localBuffer)){
                globalBuffer=new char[n+1];
                buffer=globalBuffer;
                va_list va;
                va_start(va, msg);
                vsnprintf(buffer, n+1, msg, va);
                va_end(va);
            }
            
            // %.3f with 0, causes a strange optimisation resulting in null string
            fprintf(stderr, "[Sim], %u, %.3f, %s\n", level, 0.0, buffer);
            
            if(globalBuffer){
                delete []globalBuffer;
            }
        }
        
    }
};