#version
pkg-config --modversion opencv

#run 
g++ main.cpp -o output `pkg-config --cflags --libs opencv`
