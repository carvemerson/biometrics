 g++ -ggdb `pkg-config --cflags opencv` Bibliotecas/equalization.h Bibliotecas/cvWiener2.h equalization.cpp cvWiener2.cpp main.cpp -o cv `pkg-config --libs opencv`
