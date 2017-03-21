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
CND_CONF=Release
CND_DISTDIR=dist
CND_BUILDDIR=build

# Include project Makefile
include Makefile

# Object Directory
OBJECTDIR=${CND_BUILDDIR}/${CND_CONF}/${CND_PLATFORM}

# Object Files
OBJECTFILES= \
	${OBJECTDIR}/src/main.o \
	${OBJECTDIR}/src/misc/node_factory.o \
	${OBJECTDIR}/src/nodes/basic_node.o \
	${OBJECTDIR}/src/nodes/dnn_node.o \
	${OBJECTDIR}/src/nodes/node.o \
	${OBJECTDIR}/src/tools/logging.o


# C Compiler Flags
CFLAGS=

# CC Compiler Flags
CCFLAGS=
CXXFLAGS=

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
	${LINK.cc} -o ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/asynchronous-deep-learning ${OBJECTFILES} ${LDLIBSOPTIONS}

${OBJECTDIR}/src/main.o: src/main.cpp
	${MKDIR} -p ${OBJECTDIR}/src
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/src/main.o src/main.cpp

${OBJECTDIR}/src/misc/node_factory.o: src/misc/node_factory.cpp
	${MKDIR} -p ${OBJECTDIR}/src/misc
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/src/misc/node_factory.o src/misc/node_factory.cpp

${OBJECTDIR}/src/nodes/basic_node.o: src/nodes/basic_node.cpp
	${MKDIR} -p ${OBJECTDIR}/src/nodes
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/src/nodes/basic_node.o src/nodes/basic_node.cpp

${OBJECTDIR}/src/nodes/dnn_node.o: src/nodes/dnn_node.cpp
	${MKDIR} -p ${OBJECTDIR}/src/nodes
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/src/nodes/dnn_node.o src/nodes/dnn_node.cpp

${OBJECTDIR}/src/nodes/node.o: src/nodes/node.cpp
	${MKDIR} -p ${OBJECTDIR}/src/nodes
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/src/nodes/node.o src/nodes/node.cpp

${OBJECTDIR}/src/tools/logging.o: src/tools/logging.cpp
	${MKDIR} -p ${OBJECTDIR}/src/tools
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/src/tools/logging.o src/tools/logging.cpp

# Subprojects
.build-subprojects:

# Clean Targets
.clean-conf: ${CLEAN_SUBPROJECTS}
	${RM} -r ${CND_BUILDDIR}/${CND_CONF}

# Subprojects
.clean-subprojects:

# Enable dependency checking
.dep.inc: .depcheck-impl

include .dep.inc
