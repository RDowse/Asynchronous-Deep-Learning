#
# Generated Makefile - do not edit!
#
# Edit the Makefile in the project folder instead (../Makefile). Each target
# has a -pre and a -post target defined where you can add customized code.
#
# This makefile implements configuration specific macros and targets.


# Environment
MKDIR=mkdir
CP=cp
GREP=grep
NM=nm
CCADMIN=CCadmin
RANLIB=ranlib
CC=gcc
CCC=g++
CXX=g++
FC=gfortran
AS=as

# Macros
CND_PLATFORM=GNU-Linux
CND_DLIB_EXT=so
CND_CONF=Debug
CND_DISTDIR=dist
CND_BUILDDIR=build

# Include project Makefile
include Makefile

# Object Directory
OBJECTDIR=${CND_BUILDDIR}/${CND_CONF}/${CND_PLATFORM}

# Object Files
OBJECTFILES= \
	${OBJECTDIR}/src/main.o \
	${OBJECTDIR}/src/messages/backward_propagation_message.o \
	${OBJECTDIR}/src/messages/forward_propagation_message.o \
	${OBJECTDIR}/src/misc/node_factory.o \
	${OBJECTDIR}/src/nodes/bias_node.o \
	${OBJECTDIR}/src/nodes/block_nodes/block_hidden_node.o \
	${OBJECTDIR}/src/nodes/block_nodes/block_input_node.o \
	${OBJECTDIR}/src/nodes/block_nodes/block_output_node.o \
	${OBJECTDIR}/src/nodes/hidden_node.o \
	${OBJECTDIR}/src/nodes/input_node.o \
	${OBJECTDIR}/src/nodes/node.o \
	${OBJECTDIR}/src/nodes/output_node.o \
	${OBJECTDIR}/src/nodes/sync_node.o \
	${OBJECTDIR}/src/states/backward_train_state.o \
	${OBJECTDIR}/src/states/forward_train_state.o \
	${OBJECTDIR}/src/states/predict_state.o \
	${OBJECTDIR}/src/states/state.o \
	${OBJECTDIR}/src/tools/clock.o \
	${OBJECTDIR}/src/tools/dnn_graph.o \
	${OBJECTDIR}/src/tools/logging.o \
	${OBJECTDIR}/src/training/stochastic_momentum_training.o \
	${OBJECTDIR}/src/training/stochastic_training.o

# Test Directory
TESTDIR=${CND_BUILDDIR}/${CND_CONF}/${CND_PLATFORM}/tests

# Test Files
TESTFILES= \
	${TESTDIR}/TestFiles/f1 \
	${TESTDIR}/TestFiles/f2

# Test Object Files
TESTOBJECTFILES= \
	${TESTDIR}/tests/input_node_tests.o \
	${TESTDIR}/tests/sync_node_tests.o

# C Compiler Flags
CFLAGS=

# CC Compiler Flags
CCFLAGS=-g
CXXFLAGS=-g

# Fortran Compiler Flags
FFLAGS=

# Assembler Flags
ASFLAGS=

# Link Libraries and Options
LDLIBSOPTIONS=

# Build Targets
.build-conf: ${BUILD_SUBPROJECTS}
	"${MAKE}"  -f nbproject/Makefile-${CND_CONF}.mk ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/asynchronous-deep-learning

${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/asynchronous-deep-learning: ${OBJECTFILES}
	${MKDIR} -p ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}
	${LINK.cc} -o ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/asynchronous-deep-learning ${OBJECTFILES} ${LDLIBSOPTIONS} -ltbb

${OBJECTDIR}/src/main.o: src/main.cpp
	${MKDIR} -p ${OBJECTDIR}/src
	${RM} "$@.d"
	$(COMPILE.cc) -g -Iinclude -Imnist/include -I/usr/include/eigen3 -std=c++11 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/src/main.o src/main.cpp

${OBJECTDIR}/src/messages/backward_propagation_message.o: src/messages/backward_propagation_message.cpp
	${MKDIR} -p ${OBJECTDIR}/src/messages
	${RM} "$@.d"
	$(COMPILE.cc) -g -Iinclude -Imnist/include -I/usr/include/eigen3 -std=c++11 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/src/messages/backward_propagation_message.o src/messages/backward_propagation_message.cpp

${OBJECTDIR}/src/messages/forward_propagation_message.o: src/messages/forward_propagation_message.cpp
	${MKDIR} -p ${OBJECTDIR}/src/messages
	${RM} "$@.d"
	$(COMPILE.cc) -g -Iinclude -Imnist/include -I/usr/include/eigen3 -std=c++11 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/src/messages/forward_propagation_message.o src/messages/forward_propagation_message.cpp

${OBJECTDIR}/src/misc/node_factory.o: src/misc/node_factory.cpp
	${MKDIR} -p ${OBJECTDIR}/src/misc
	${RM} "$@.d"
	$(COMPILE.cc) -g -Iinclude -Imnist/include -I/usr/include/eigen3 -std=c++11 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/src/misc/node_factory.o src/misc/node_factory.cpp

${OBJECTDIR}/src/nodes/bias_node.o: src/nodes/bias_node.cpp
	${MKDIR} -p ${OBJECTDIR}/src/nodes
	${RM} "$@.d"
	$(COMPILE.cc) -g -Iinclude -Imnist/include -I/usr/include/eigen3 -std=c++11 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/src/nodes/bias_node.o src/nodes/bias_node.cpp

${OBJECTDIR}/src/nodes/block_nodes/block_hidden_node.o: src/nodes/block_nodes/block_hidden_node.cpp
	${MKDIR} -p ${OBJECTDIR}/src/nodes/block_nodes
	${RM} "$@.d"
	$(COMPILE.cc) -g -Iinclude -Imnist/include -I/usr/include/eigen3 -std=c++11 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/src/nodes/block_nodes/block_hidden_node.o src/nodes/block_nodes/block_hidden_node.cpp

${OBJECTDIR}/src/nodes/block_nodes/block_input_node.o: src/nodes/block_nodes/block_input_node.cpp
	${MKDIR} -p ${OBJECTDIR}/src/nodes/block_nodes
	${RM} "$@.d"
	$(COMPILE.cc) -g -Iinclude -Imnist/include -I/usr/include/eigen3 -std=c++11 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/src/nodes/block_nodes/block_input_node.o src/nodes/block_nodes/block_input_node.cpp

${OBJECTDIR}/src/nodes/block_nodes/block_output_node.o: src/nodes/block_nodes/block_output_node.cpp
	${MKDIR} -p ${OBJECTDIR}/src/nodes/block_nodes
	${RM} "$@.d"
	$(COMPILE.cc) -g -Iinclude -Imnist/include -I/usr/include/eigen3 -std=c++11 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/src/nodes/block_nodes/block_output_node.o src/nodes/block_nodes/block_output_node.cpp

${OBJECTDIR}/src/nodes/hidden_node.o: src/nodes/hidden_node.cpp
	${MKDIR} -p ${OBJECTDIR}/src/nodes
	${RM} "$@.d"
	$(COMPILE.cc) -g -Iinclude -Imnist/include -I/usr/include/eigen3 -std=c++11 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/src/nodes/hidden_node.o src/nodes/hidden_node.cpp

${OBJECTDIR}/src/nodes/input_node.o: src/nodes/input_node.cpp
	${MKDIR} -p ${OBJECTDIR}/src/nodes
	${RM} "$@.d"
	$(COMPILE.cc) -g -Iinclude -Imnist/include -I/usr/include/eigen3 -std=c++11 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/src/nodes/input_node.o src/nodes/input_node.cpp

${OBJECTDIR}/src/nodes/node.o: src/nodes/node.cpp
	${MKDIR} -p ${OBJECTDIR}/src/nodes
	${RM} "$@.d"
	$(COMPILE.cc) -g -Iinclude -Imnist/include -I/usr/include/eigen3 -std=c++11 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/src/nodes/node.o src/nodes/node.cpp

${OBJECTDIR}/src/nodes/output_node.o: src/nodes/output_node.cpp
	${MKDIR} -p ${OBJECTDIR}/src/nodes
	${RM} "$@.d"
	$(COMPILE.cc) -g -Iinclude -Imnist/include -I/usr/include/eigen3 -std=c++11 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/src/nodes/output_node.o src/nodes/output_node.cpp

${OBJECTDIR}/src/nodes/sync_node.o: src/nodes/sync_node.cpp
	${MKDIR} -p ${OBJECTDIR}/src/nodes
	${RM} "$@.d"
	$(COMPILE.cc) -g -Iinclude -Imnist/include -I/usr/include/eigen3 -std=c++11 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/src/nodes/sync_node.o src/nodes/sync_node.cpp

${OBJECTDIR}/src/states/backward_train_state.o: src/states/backward_train_state.cpp
	${MKDIR} -p ${OBJECTDIR}/src/states
	${RM} "$@.d"
	$(COMPILE.cc) -g -Iinclude -Imnist/include -I/usr/include/eigen3 -std=c++11 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/src/states/backward_train_state.o src/states/backward_train_state.cpp

${OBJECTDIR}/src/states/forward_train_state.o: src/states/forward_train_state.cpp
	${MKDIR} -p ${OBJECTDIR}/src/states
	${RM} "$@.d"
	$(COMPILE.cc) -g -Iinclude -Imnist/include -I/usr/include/eigen3 -std=c++11 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/src/states/forward_train_state.o src/states/forward_train_state.cpp

${OBJECTDIR}/src/states/predict_state.o: src/states/predict_state.cpp
	${MKDIR} -p ${OBJECTDIR}/src/states
	${RM} "$@.d"
	$(COMPILE.cc) -g -Iinclude -Imnist/include -I/usr/include/eigen3 -std=c++11 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/src/states/predict_state.o src/states/predict_state.cpp

${OBJECTDIR}/src/states/state.o: src/states/state.cpp
	${MKDIR} -p ${OBJECTDIR}/src/states
	${RM} "$@.d"
	$(COMPILE.cc) -g -Iinclude -Imnist/include -I/usr/include/eigen3 -std=c++11 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/src/states/state.o src/states/state.cpp

${OBJECTDIR}/src/tools/clock.o: src/tools/clock.cpp
	${MKDIR} -p ${OBJECTDIR}/src/tools
	${RM} "$@.d"
	$(COMPILE.cc) -g -Iinclude -Imnist/include -I/usr/include/eigen3 -std=c++11 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/src/tools/clock.o src/tools/clock.cpp

${OBJECTDIR}/src/tools/dnn_graph.o: src/tools/dnn_graph.cpp
	${MKDIR} -p ${OBJECTDIR}/src/tools
	${RM} "$@.d"
	$(COMPILE.cc) -g -Iinclude -Imnist/include -I/usr/include/eigen3 -std=c++11 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/src/tools/dnn_graph.o src/tools/dnn_graph.cpp

${OBJECTDIR}/src/tools/logging.o: src/tools/logging.cpp
	${MKDIR} -p ${OBJECTDIR}/src/tools
	${RM} "$@.d"
	$(COMPILE.cc) -g -Iinclude -Imnist/include -I/usr/include/eigen3 -std=c++11 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/src/tools/logging.o src/tools/logging.cpp

${OBJECTDIR}/src/training/stochastic_momentum_training.o: src/training/stochastic_momentum_training.cpp
	${MKDIR} -p ${OBJECTDIR}/src/training
	${RM} "$@.d"
	$(COMPILE.cc) -g -Iinclude -Imnist/include -I/usr/include/eigen3 -std=c++11 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/src/training/stochastic_momentum_training.o src/training/stochastic_momentum_training.cpp

${OBJECTDIR}/src/training/stochastic_training.o: src/training/stochastic_training.cpp
	${MKDIR} -p ${OBJECTDIR}/src/training
	${RM} "$@.d"
	$(COMPILE.cc) -g -Iinclude -Imnist/include -I/usr/include/eigen3 -std=c++11 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/src/training/stochastic_training.o src/training/stochastic_training.cpp

# Subprojects
.build-subprojects:

# Build Test Targets
.build-tests-conf: .build-tests-subprojects .build-conf ${TESTFILES}
.build-tests-subprojects:

${TESTDIR}/TestFiles/f1: ${TESTDIR}/tests/input_node_tests.o ${OBJECTFILES:%.o=%_nomain.o}
	${MKDIR} -p ${TESTDIR}/TestFiles
	${LINK.cc} -o ${TESTDIR}/TestFiles/f1 $^ ${LDLIBSOPTIONS}   

${TESTDIR}/TestFiles/f2: ${TESTDIR}/tests/sync_node_tests.o ${OBJECTFILES:%.o=%_nomain.o}
	${MKDIR} -p ${TESTDIR}/TestFiles
	${LINK.cc} -o ${TESTDIR}/TestFiles/f2 $^ ${LDLIBSOPTIONS}   


${TESTDIR}/tests/input_node_tests.o: tests/input_node_tests.cpp 
	${MKDIR} -p ${TESTDIR}/tests
	${RM} "$@.d"
	$(COMPILE.cc) -g -Iinclude -Imnist/include -I/usr/include/eigen3 -I. -std=c++11 -MMD -MP -MF "$@.d" -o ${TESTDIR}/tests/input_node_tests.o tests/input_node_tests.cpp


${TESTDIR}/tests/sync_node_tests.o: tests/sync_node_tests.cpp 
	${MKDIR} -p ${TESTDIR}/tests
	${RM} "$@.d"
	$(COMPILE.cc) -g -Iinclude -Imnist/include -I/usr/include/eigen3 -I. -std=c++11 -MMD -MP -MF "$@.d" -o ${TESTDIR}/tests/sync_node_tests.o tests/sync_node_tests.cpp


${OBJECTDIR}/src/main_nomain.o: ${OBJECTDIR}/src/main.o src/main.cpp 
	${MKDIR} -p ${OBJECTDIR}/src
	@NMOUTPUT=`${NM} ${OBJECTDIR}/src/main.o`; \
	if (echo "$$NMOUTPUT" | ${GREP} '|main$$') || \
	   (echo "$$NMOUTPUT" | ${GREP} 'T main$$') || \
	   (echo "$$NMOUTPUT" | ${GREP} 'T _main$$'); \
	then  \
	    ${RM} "$@.d";\
	    $(COMPILE.cc) -g -Iinclude -Imnist/include -I/usr/include/eigen3 -std=c++11 -Dmain=__nomain -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/src/main_nomain.o src/main.cpp;\
	else  \
	    ${CP} ${OBJECTDIR}/src/main.o ${OBJECTDIR}/src/main_nomain.o;\
	fi

${OBJECTDIR}/src/messages/backward_propagation_message_nomain.o: ${OBJECTDIR}/src/messages/backward_propagation_message.o src/messages/backward_propagation_message.cpp 
	${MKDIR} -p ${OBJECTDIR}/src/messages
	@NMOUTPUT=`${NM} ${OBJECTDIR}/src/messages/backward_propagation_message.o`; \
	if (echo "$$NMOUTPUT" | ${GREP} '|main$$') || \
	   (echo "$$NMOUTPUT" | ${GREP} 'T main$$') || \
	   (echo "$$NMOUTPUT" | ${GREP} 'T _main$$'); \
	then  \
	    ${RM} "$@.d";\
	    $(COMPILE.cc) -g -Iinclude -Imnist/include -I/usr/include/eigen3 -std=c++11 -Dmain=__nomain -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/src/messages/backward_propagation_message_nomain.o src/messages/backward_propagation_message.cpp;\
	else  \
	    ${CP} ${OBJECTDIR}/src/messages/backward_propagation_message.o ${OBJECTDIR}/src/messages/backward_propagation_message_nomain.o;\
	fi

${OBJECTDIR}/src/messages/forward_propagation_message_nomain.o: ${OBJECTDIR}/src/messages/forward_propagation_message.o src/messages/forward_propagation_message.cpp 
	${MKDIR} -p ${OBJECTDIR}/src/messages
	@NMOUTPUT=`${NM} ${OBJECTDIR}/src/messages/forward_propagation_message.o`; \
	if (echo "$$NMOUTPUT" | ${GREP} '|main$$') || \
	   (echo "$$NMOUTPUT" | ${GREP} 'T main$$') || \
	   (echo "$$NMOUTPUT" | ${GREP} 'T _main$$'); \
	then  \
	    ${RM} "$@.d";\
	    $(COMPILE.cc) -g -Iinclude -Imnist/include -I/usr/include/eigen3 -std=c++11 -Dmain=__nomain -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/src/messages/forward_propagation_message_nomain.o src/messages/forward_propagation_message.cpp;\
	else  \
	    ${CP} ${OBJECTDIR}/src/messages/forward_propagation_message.o ${OBJECTDIR}/src/messages/forward_propagation_message_nomain.o;\
	fi

${OBJECTDIR}/src/misc/node_factory_nomain.o: ${OBJECTDIR}/src/misc/node_factory.o src/misc/node_factory.cpp 
	${MKDIR} -p ${OBJECTDIR}/src/misc
	@NMOUTPUT=`${NM} ${OBJECTDIR}/src/misc/node_factory.o`; \
	if (echo "$$NMOUTPUT" | ${GREP} '|main$$') || \
	   (echo "$$NMOUTPUT" | ${GREP} 'T main$$') || \
	   (echo "$$NMOUTPUT" | ${GREP} 'T _main$$'); \
	then  \
	    ${RM} "$@.d";\
	    $(COMPILE.cc) -g -Iinclude -Imnist/include -I/usr/include/eigen3 -std=c++11 -Dmain=__nomain -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/src/misc/node_factory_nomain.o src/misc/node_factory.cpp;\
	else  \
	    ${CP} ${OBJECTDIR}/src/misc/node_factory.o ${OBJECTDIR}/src/misc/node_factory_nomain.o;\
	fi

${OBJECTDIR}/src/nodes/bias_node_nomain.o: ${OBJECTDIR}/src/nodes/bias_node.o src/nodes/bias_node.cpp 
	${MKDIR} -p ${OBJECTDIR}/src/nodes
	@NMOUTPUT=`${NM} ${OBJECTDIR}/src/nodes/bias_node.o`; \
	if (echo "$$NMOUTPUT" | ${GREP} '|main$$') || \
	   (echo "$$NMOUTPUT" | ${GREP} 'T main$$') || \
	   (echo "$$NMOUTPUT" | ${GREP} 'T _main$$'); \
	then  \
	    ${RM} "$@.d";\
	    $(COMPILE.cc) -g -Iinclude -Imnist/include -I/usr/include/eigen3 -std=c++11 -Dmain=__nomain -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/src/nodes/bias_node_nomain.o src/nodes/bias_node.cpp;\
	else  \
	    ${CP} ${OBJECTDIR}/src/nodes/bias_node.o ${OBJECTDIR}/src/nodes/bias_node_nomain.o;\
	fi

${OBJECTDIR}/src/nodes/block_nodes/block_hidden_node_nomain.o: ${OBJECTDIR}/src/nodes/block_nodes/block_hidden_node.o src/nodes/block_nodes/block_hidden_node.cpp 
	${MKDIR} -p ${OBJECTDIR}/src/nodes/block_nodes
	@NMOUTPUT=`${NM} ${OBJECTDIR}/src/nodes/block_nodes/block_hidden_node.o`; \
	if (echo "$$NMOUTPUT" | ${GREP} '|main$$') || \
	   (echo "$$NMOUTPUT" | ${GREP} 'T main$$') || \
	   (echo "$$NMOUTPUT" | ${GREP} 'T _main$$'); \
	then  \
	    ${RM} "$@.d";\
	    $(COMPILE.cc) -g -Iinclude -Imnist/include -I/usr/include/eigen3 -std=c++11 -Dmain=__nomain -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/src/nodes/block_nodes/block_hidden_node_nomain.o src/nodes/block_nodes/block_hidden_node.cpp;\
	else  \
	    ${CP} ${OBJECTDIR}/src/nodes/block_nodes/block_hidden_node.o ${OBJECTDIR}/src/nodes/block_nodes/block_hidden_node_nomain.o;\
	fi

${OBJECTDIR}/src/nodes/block_nodes/block_input_node_nomain.o: ${OBJECTDIR}/src/nodes/block_nodes/block_input_node.o src/nodes/block_nodes/block_input_node.cpp 
	${MKDIR} -p ${OBJECTDIR}/src/nodes/block_nodes
	@NMOUTPUT=`${NM} ${OBJECTDIR}/src/nodes/block_nodes/block_input_node.o`; \
	if (echo "$$NMOUTPUT" | ${GREP} '|main$$') || \
	   (echo "$$NMOUTPUT" | ${GREP} 'T main$$') || \
	   (echo "$$NMOUTPUT" | ${GREP} 'T _main$$'); \
	then  \
	    ${RM} "$@.d";\
	    $(COMPILE.cc) -g -Iinclude -Imnist/include -I/usr/include/eigen3 -std=c++11 -Dmain=__nomain -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/src/nodes/block_nodes/block_input_node_nomain.o src/nodes/block_nodes/block_input_node.cpp;\
	else  \
	    ${CP} ${OBJECTDIR}/src/nodes/block_nodes/block_input_node.o ${OBJECTDIR}/src/nodes/block_nodes/block_input_node_nomain.o;\
	fi

${OBJECTDIR}/src/nodes/block_nodes/block_output_node_nomain.o: ${OBJECTDIR}/src/nodes/block_nodes/block_output_node.o src/nodes/block_nodes/block_output_node.cpp 
	${MKDIR} -p ${OBJECTDIR}/src/nodes/block_nodes
	@NMOUTPUT=`${NM} ${OBJECTDIR}/src/nodes/block_nodes/block_output_node.o`; \
	if (echo "$$NMOUTPUT" | ${GREP} '|main$$') || \
	   (echo "$$NMOUTPUT" | ${GREP} 'T main$$') || \
	   (echo "$$NMOUTPUT" | ${GREP} 'T _main$$'); \
	then  \
	    ${RM} "$@.d";\
	    $(COMPILE.cc) -g -Iinclude -Imnist/include -I/usr/include/eigen3 -std=c++11 -Dmain=__nomain -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/src/nodes/block_nodes/block_output_node_nomain.o src/nodes/block_nodes/block_output_node.cpp;\
	else  \
	    ${CP} ${OBJECTDIR}/src/nodes/block_nodes/block_output_node.o ${OBJECTDIR}/src/nodes/block_nodes/block_output_node_nomain.o;\
	fi

${OBJECTDIR}/src/nodes/hidden_node_nomain.o: ${OBJECTDIR}/src/nodes/hidden_node.o src/nodes/hidden_node.cpp 
	${MKDIR} -p ${OBJECTDIR}/src/nodes
	@NMOUTPUT=`${NM} ${OBJECTDIR}/src/nodes/hidden_node.o`; \
	if (echo "$$NMOUTPUT" | ${GREP} '|main$$') || \
	   (echo "$$NMOUTPUT" | ${GREP} 'T main$$') || \
	   (echo "$$NMOUTPUT" | ${GREP} 'T _main$$'); \
	then  \
	    ${RM} "$@.d";\
	    $(COMPILE.cc) -g -Iinclude -Imnist/include -I/usr/include/eigen3 -std=c++11 -Dmain=__nomain -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/src/nodes/hidden_node_nomain.o src/nodes/hidden_node.cpp;\
	else  \
	    ${CP} ${OBJECTDIR}/src/nodes/hidden_node.o ${OBJECTDIR}/src/nodes/hidden_node_nomain.o;\
	fi

${OBJECTDIR}/src/nodes/input_node_nomain.o: ${OBJECTDIR}/src/nodes/input_node.o src/nodes/input_node.cpp 
	${MKDIR} -p ${OBJECTDIR}/src/nodes
	@NMOUTPUT=`${NM} ${OBJECTDIR}/src/nodes/input_node.o`; \
	if (echo "$$NMOUTPUT" | ${GREP} '|main$$') || \
	   (echo "$$NMOUTPUT" | ${GREP} 'T main$$') || \
	   (echo "$$NMOUTPUT" | ${GREP} 'T _main$$'); \
	then  \
	    ${RM} "$@.d";\
	    $(COMPILE.cc) -g -Iinclude -Imnist/include -I/usr/include/eigen3 -std=c++11 -Dmain=__nomain -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/src/nodes/input_node_nomain.o src/nodes/input_node.cpp;\
	else  \
	    ${CP} ${OBJECTDIR}/src/nodes/input_node.o ${OBJECTDIR}/src/nodes/input_node_nomain.o;\
	fi

${OBJECTDIR}/src/nodes/node_nomain.o: ${OBJECTDIR}/src/nodes/node.o src/nodes/node.cpp 
	${MKDIR} -p ${OBJECTDIR}/src/nodes
	@NMOUTPUT=`${NM} ${OBJECTDIR}/src/nodes/node.o`; \
	if (echo "$$NMOUTPUT" | ${GREP} '|main$$') || \
	   (echo "$$NMOUTPUT" | ${GREP} 'T main$$') || \
	   (echo "$$NMOUTPUT" | ${GREP} 'T _main$$'); \
	then  \
	    ${RM} "$@.d";\
	    $(COMPILE.cc) -g -Iinclude -Imnist/include -I/usr/include/eigen3 -std=c++11 -Dmain=__nomain -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/src/nodes/node_nomain.o src/nodes/node.cpp;\
	else  \
	    ${CP} ${OBJECTDIR}/src/nodes/node.o ${OBJECTDIR}/src/nodes/node_nomain.o;\
	fi

${OBJECTDIR}/src/nodes/output_node_nomain.o: ${OBJECTDIR}/src/nodes/output_node.o src/nodes/output_node.cpp 
	${MKDIR} -p ${OBJECTDIR}/src/nodes
	@NMOUTPUT=`${NM} ${OBJECTDIR}/src/nodes/output_node.o`; \
	if (echo "$$NMOUTPUT" | ${GREP} '|main$$') || \
	   (echo "$$NMOUTPUT" | ${GREP} 'T main$$') || \
	   (echo "$$NMOUTPUT" | ${GREP} 'T _main$$'); \
	then  \
	    ${RM} "$@.d";\
	    $(COMPILE.cc) -g -Iinclude -Imnist/include -I/usr/include/eigen3 -std=c++11 -Dmain=__nomain -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/src/nodes/output_node_nomain.o src/nodes/output_node.cpp;\
	else  \
	    ${CP} ${OBJECTDIR}/src/nodes/output_node.o ${OBJECTDIR}/src/nodes/output_node_nomain.o;\
	fi

${OBJECTDIR}/src/nodes/sync_node_nomain.o: ${OBJECTDIR}/src/nodes/sync_node.o src/nodes/sync_node.cpp 
	${MKDIR} -p ${OBJECTDIR}/src/nodes
	@NMOUTPUT=`${NM} ${OBJECTDIR}/src/nodes/sync_node.o`; \
	if (echo "$$NMOUTPUT" | ${GREP} '|main$$') || \
	   (echo "$$NMOUTPUT" | ${GREP} 'T main$$') || \
	   (echo "$$NMOUTPUT" | ${GREP} 'T _main$$'); \
	then  \
	    ${RM} "$@.d";\
	    $(COMPILE.cc) -g -Iinclude -Imnist/include -I/usr/include/eigen3 -std=c++11 -Dmain=__nomain -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/src/nodes/sync_node_nomain.o src/nodes/sync_node.cpp;\
	else  \
	    ${CP} ${OBJECTDIR}/src/nodes/sync_node.o ${OBJECTDIR}/src/nodes/sync_node_nomain.o;\
	fi

${OBJECTDIR}/src/states/backward_train_state_nomain.o: ${OBJECTDIR}/src/states/backward_train_state.o src/states/backward_train_state.cpp 
	${MKDIR} -p ${OBJECTDIR}/src/states
	@NMOUTPUT=`${NM} ${OBJECTDIR}/src/states/backward_train_state.o`; \
	if (echo "$$NMOUTPUT" | ${GREP} '|main$$') || \
	   (echo "$$NMOUTPUT" | ${GREP} 'T main$$') || \
	   (echo "$$NMOUTPUT" | ${GREP} 'T _main$$'); \
	then  \
	    ${RM} "$@.d";\
	    $(COMPILE.cc) -g -Iinclude -Imnist/include -I/usr/include/eigen3 -std=c++11 -Dmain=__nomain -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/src/states/backward_train_state_nomain.o src/states/backward_train_state.cpp;\
	else  \
	    ${CP} ${OBJECTDIR}/src/states/backward_train_state.o ${OBJECTDIR}/src/states/backward_train_state_nomain.o;\
	fi

${OBJECTDIR}/src/states/forward_train_state_nomain.o: ${OBJECTDIR}/src/states/forward_train_state.o src/states/forward_train_state.cpp 
	${MKDIR} -p ${OBJECTDIR}/src/states
	@NMOUTPUT=`${NM} ${OBJECTDIR}/src/states/forward_train_state.o`; \
	if (echo "$$NMOUTPUT" | ${GREP} '|main$$') || \
	   (echo "$$NMOUTPUT" | ${GREP} 'T main$$') || \
	   (echo "$$NMOUTPUT" | ${GREP} 'T _main$$'); \
	then  \
	    ${RM} "$@.d";\
	    $(COMPILE.cc) -g -Iinclude -Imnist/include -I/usr/include/eigen3 -std=c++11 -Dmain=__nomain -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/src/states/forward_train_state_nomain.o src/states/forward_train_state.cpp;\
	else  \
	    ${CP} ${OBJECTDIR}/src/states/forward_train_state.o ${OBJECTDIR}/src/states/forward_train_state_nomain.o;\
	fi

${OBJECTDIR}/src/states/predict_state_nomain.o: ${OBJECTDIR}/src/states/predict_state.o src/states/predict_state.cpp 
	${MKDIR} -p ${OBJECTDIR}/src/states
	@NMOUTPUT=`${NM} ${OBJECTDIR}/src/states/predict_state.o`; \
	if (echo "$$NMOUTPUT" | ${GREP} '|main$$') || \
	   (echo "$$NMOUTPUT" | ${GREP} 'T main$$') || \
	   (echo "$$NMOUTPUT" | ${GREP} 'T _main$$'); \
	then  \
	    ${RM} "$@.d";\
	    $(COMPILE.cc) -g -Iinclude -Imnist/include -I/usr/include/eigen3 -std=c++11 -Dmain=__nomain -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/src/states/predict_state_nomain.o src/states/predict_state.cpp;\
	else  \
	    ${CP} ${OBJECTDIR}/src/states/predict_state.o ${OBJECTDIR}/src/states/predict_state_nomain.o;\
	fi

${OBJECTDIR}/src/states/state_nomain.o: ${OBJECTDIR}/src/states/state.o src/states/state.cpp 
	${MKDIR} -p ${OBJECTDIR}/src/states
	@NMOUTPUT=`${NM} ${OBJECTDIR}/src/states/state.o`; \
	if (echo "$$NMOUTPUT" | ${GREP} '|main$$') || \
	   (echo "$$NMOUTPUT" | ${GREP} 'T main$$') || \
	   (echo "$$NMOUTPUT" | ${GREP} 'T _main$$'); \
	then  \
	    ${RM} "$@.d";\
	    $(COMPILE.cc) -g -Iinclude -Imnist/include -I/usr/include/eigen3 -std=c++11 -Dmain=__nomain -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/src/states/state_nomain.o src/states/state.cpp;\
	else  \
	    ${CP} ${OBJECTDIR}/src/states/state.o ${OBJECTDIR}/src/states/state_nomain.o;\
	fi

${OBJECTDIR}/src/tools/clock_nomain.o: ${OBJECTDIR}/src/tools/clock.o src/tools/clock.cpp 
	${MKDIR} -p ${OBJECTDIR}/src/tools
	@NMOUTPUT=`${NM} ${OBJECTDIR}/src/tools/clock.o`; \
	if (echo "$$NMOUTPUT" | ${GREP} '|main$$') || \
	   (echo "$$NMOUTPUT" | ${GREP} 'T main$$') || \
	   (echo "$$NMOUTPUT" | ${GREP} 'T _main$$'); \
	then  \
	    ${RM} "$@.d";\
	    $(COMPILE.cc) -g -Iinclude -Imnist/include -I/usr/include/eigen3 -std=c++11 -Dmain=__nomain -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/src/tools/clock_nomain.o src/tools/clock.cpp;\
	else  \
	    ${CP} ${OBJECTDIR}/src/tools/clock.o ${OBJECTDIR}/src/tools/clock_nomain.o;\
	fi

${OBJECTDIR}/src/tools/dnn_graph_nomain.o: ${OBJECTDIR}/src/tools/dnn_graph.o src/tools/dnn_graph.cpp 
	${MKDIR} -p ${OBJECTDIR}/src/tools
	@NMOUTPUT=`${NM} ${OBJECTDIR}/src/tools/dnn_graph.o`; \
	if (echo "$$NMOUTPUT" | ${GREP} '|main$$') || \
	   (echo "$$NMOUTPUT" | ${GREP} 'T main$$') || \
	   (echo "$$NMOUTPUT" | ${GREP} 'T _main$$'); \
	then  \
	    ${RM} "$@.d";\
	    $(COMPILE.cc) -g -Iinclude -Imnist/include -I/usr/include/eigen3 -std=c++11 -Dmain=__nomain -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/src/tools/dnn_graph_nomain.o src/tools/dnn_graph.cpp;\
	else  \
	    ${CP} ${OBJECTDIR}/src/tools/dnn_graph.o ${OBJECTDIR}/src/tools/dnn_graph_nomain.o;\
	fi

${OBJECTDIR}/src/tools/logging_nomain.o: ${OBJECTDIR}/src/tools/logging.o src/tools/logging.cpp 
	${MKDIR} -p ${OBJECTDIR}/src/tools
	@NMOUTPUT=`${NM} ${OBJECTDIR}/src/tools/logging.o`; \
	if (echo "$$NMOUTPUT" | ${GREP} '|main$$') || \
	   (echo "$$NMOUTPUT" | ${GREP} 'T main$$') || \
	   (echo "$$NMOUTPUT" | ${GREP} 'T _main$$'); \
	then  \
	    ${RM} "$@.d";\
	    $(COMPILE.cc) -g -Iinclude -Imnist/include -I/usr/include/eigen3 -std=c++11 -Dmain=__nomain -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/src/tools/logging_nomain.o src/tools/logging.cpp;\
	else  \
	    ${CP} ${OBJECTDIR}/src/tools/logging.o ${OBJECTDIR}/src/tools/logging_nomain.o;\
	fi

${OBJECTDIR}/src/training/stochastic_momentum_training_nomain.o: ${OBJECTDIR}/src/training/stochastic_momentum_training.o src/training/stochastic_momentum_training.cpp 
	${MKDIR} -p ${OBJECTDIR}/src/training
	@NMOUTPUT=`${NM} ${OBJECTDIR}/src/training/stochastic_momentum_training.o`; \
	if (echo "$$NMOUTPUT" | ${GREP} '|main$$') || \
	   (echo "$$NMOUTPUT" | ${GREP} 'T main$$') || \
	   (echo "$$NMOUTPUT" | ${GREP} 'T _main$$'); \
	then  \
	    ${RM} "$@.d";\
	    $(COMPILE.cc) -g -Iinclude -Imnist/include -I/usr/include/eigen3 -std=c++11 -Dmain=__nomain -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/src/training/stochastic_momentum_training_nomain.o src/training/stochastic_momentum_training.cpp;\
	else  \
	    ${CP} ${OBJECTDIR}/src/training/stochastic_momentum_training.o ${OBJECTDIR}/src/training/stochastic_momentum_training_nomain.o;\
	fi

${OBJECTDIR}/src/training/stochastic_training_nomain.o: ${OBJECTDIR}/src/training/stochastic_training.o src/training/stochastic_training.cpp 
	${MKDIR} -p ${OBJECTDIR}/src/training
	@NMOUTPUT=`${NM} ${OBJECTDIR}/src/training/stochastic_training.o`; \
	if (echo "$$NMOUTPUT" | ${GREP} '|main$$') || \
	   (echo "$$NMOUTPUT" | ${GREP} 'T main$$') || \
	   (echo "$$NMOUTPUT" | ${GREP} 'T _main$$'); \
	then  \
	    ${RM} "$@.d";\
	    $(COMPILE.cc) -g -Iinclude -Imnist/include -I/usr/include/eigen3 -std=c++11 -Dmain=__nomain -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/src/training/stochastic_training_nomain.o src/training/stochastic_training.cpp;\
	else  \
	    ${CP} ${OBJECTDIR}/src/training/stochastic_training.o ${OBJECTDIR}/src/training/stochastic_training_nomain.o;\
	fi

# Run Test Targets
.test-conf:
	@if [ "${TEST}" = "" ]; \
	then  \
	    ${TESTDIR}/TestFiles/f1 || true; \
	    ${TESTDIR}/TestFiles/f2 || true; \
	else  \
	    ./${TEST} || true; \
	fi

# Clean Targets
.clean-conf: ${CLEAN_SUBPROJECTS}
	${RM} -r ${CND_BUILDDIR}/${CND_CONF}

# Subprojects
.clean-subprojects:

# Enable dependency checking
.dep.inc: .depcheck-impl

include .dep.inc
