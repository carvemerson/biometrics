 g++ -ggdb `pkg-config --cflags opencv` equalization.h cvWiener2.h equalization.cpp cvWiener2.cpp test_wiener.cpp -o cv `pkg-config --libs opencv`
