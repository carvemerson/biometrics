 g++ -ggdb `pkg-config --cflags opencv` Bibliotecas/equalization.h Bibliotecas/cvWiener2.h Bibliotecas/thinning.h equalization.cpp cvWiener2.cpp thinning.cpp main.cpp -o cv `pkg-config --libs opencv`
