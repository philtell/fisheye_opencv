# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/philtell/CLionProjects/fisheye

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/philtell/CLionProjects/fisheye/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/fisheye.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/fisheye.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/fisheye.dir/flags.make

CMakeFiles/fisheye.dir/main.cpp.o: CMakeFiles/fisheye.dir/flags.make
CMakeFiles/fisheye.dir/main.cpp.o: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/philtell/CLionProjects/fisheye/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/fisheye.dir/main.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/fisheye.dir/main.cpp.o -c /home/philtell/CLionProjects/fisheye/main.cpp

CMakeFiles/fisheye.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/fisheye.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/philtell/CLionProjects/fisheye/main.cpp > CMakeFiles/fisheye.dir/main.cpp.i

CMakeFiles/fisheye.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/fisheye.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/philtell/CLionProjects/fisheye/main.cpp -o CMakeFiles/fisheye.dir/main.cpp.s

CMakeFiles/fisheye.dir/main.cpp.o.requires:

.PHONY : CMakeFiles/fisheye.dir/main.cpp.o.requires

CMakeFiles/fisheye.dir/main.cpp.o.provides: CMakeFiles/fisheye.dir/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/fisheye.dir/build.make CMakeFiles/fisheye.dir/main.cpp.o.provides.build
.PHONY : CMakeFiles/fisheye.dir/main.cpp.o.provides

CMakeFiles/fisheye.dir/main.cpp.o.provides.build: CMakeFiles/fisheye.dir/main.cpp.o


# Object files for target fisheye
fisheye_OBJECTS = \
"CMakeFiles/fisheye.dir/main.cpp.o"

# External object files for target fisheye
fisheye_EXTERNAL_OBJECTS =

fisheye: CMakeFiles/fisheye.dir/main.cpp.o
fisheye: CMakeFiles/fisheye.dir/build.make
fisheye: /usr/local/lib/libopencv_dnn.so.4.3.0
fisheye: /usr/local/lib/libopencv_gapi.so.4.3.0
fisheye: /usr/local/lib/libopencv_highgui.so.4.3.0
fisheye: /usr/local/lib/libopencv_ml.so.4.3.0
fisheye: /usr/local/lib/libopencv_objdetect.so.4.3.0
fisheye: /usr/local/lib/libopencv_photo.so.4.3.0
fisheye: /usr/local/lib/libopencv_stitching.so.4.3.0
fisheye: /usr/local/lib/libopencv_video.so.4.3.0
fisheye: /usr/local/lib/libopencv_videoio.so.4.3.0
fisheye: /usr/local/lib/libopencv_imgcodecs.so.4.3.0
fisheye: /usr/local/lib/libopencv_calib3d.so.4.3.0
fisheye: /usr/local/lib/libopencv_features2d.so.4.3.0
fisheye: /usr/local/lib/libopencv_flann.so.4.3.0
fisheye: /usr/local/lib/libopencv_imgproc.so.4.3.0
fisheye: /usr/local/lib/libopencv_core.so.4.3.0
fisheye: CMakeFiles/fisheye.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/philtell/CLionProjects/fisheye/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable fisheye"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/fisheye.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/fisheye.dir/build: fisheye

.PHONY : CMakeFiles/fisheye.dir/build

CMakeFiles/fisheye.dir/requires: CMakeFiles/fisheye.dir/main.cpp.o.requires

.PHONY : CMakeFiles/fisheye.dir/requires

CMakeFiles/fisheye.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/fisheye.dir/cmake_clean.cmake
.PHONY : CMakeFiles/fisheye.dir/clean

CMakeFiles/fisheye.dir/depend:
	cd /home/philtell/CLionProjects/fisheye/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/philtell/CLionProjects/fisheye /home/philtell/CLionProjects/fisheye /home/philtell/CLionProjects/fisheye/cmake-build-debug /home/philtell/CLionProjects/fisheye/cmake-build-debug /home/philtell/CLionProjects/fisheye/cmake-build-debug/CMakeFiles/fisheye.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/fisheye.dir/depend

