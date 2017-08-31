cmake_minimum_required(VERSION 2.8)
project(HumanCounter)

set(NAME ${PROJECT_NAME})

set(CMAKE_CONFIGURATION_TYPES Debug Release)

include_directories("${CMAKE_CURRENT_LIST_DIR}/include")

file(GLOB SRC "${CMAKE_CURRENT_LIST_DIR}/*.cpp")


if(CMAKE_COMPILER_IS_GNUCXX)
	set(CMAKE_CXX_FLAGS " ${CMAKE_CXX_FLAGS} -std=gnu++11 -fpermissive -fexceptions -fopenmp")
	set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -g")
	set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O3 -DNDEBUG")
endif()

find_package(OpenCV REQUIRED)
message(STATUS "*${NAME}* OpenCV include directory: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "*${NAME}* OpenCV librairies: ${OpenCV_LIBS}")

include_directories("${OpenCV_INCLUDE_DIRS}")

find_package(Boost 1.55 REQUIRED
	COMPONENTS
		program_options
		system
		filesystem
)

include_directories("${Boost_INCLUDE_DIRS}")
message(STATUS "*${NAME}* Boost include directory: ${Boost_INCLUDE_DIRS}")
message(STATUS "*${NAME}* Boost librairies: ${Boost_LIBRARIES}")

#add_library(obj OBJECT ${src})

if(CMAKE_SYSTEM_NAME MATCHES Linux)
	add_executable( ${NAME} ${SRC})
	set_target_properties( ${NAME} PROPERTIES LINKER_LANGUAGE CXX DEBUG_POSTFIX d)
	target_link_libraries ( ${NAME}
		${OpenCV_LIBS}
		${Boost_LIBRARIES}
        #raspicam
		#raspicam_cv
	)
endif()

install(
	TARGETS ${NAME}
	RUNTIME DESTINATION bin
)
