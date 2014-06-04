#include <algorithm>
#include <string>
#include <vector>
#include <cmath>
#include <climits>
#include <cstdio>
#include <cstring>

#include "Bibliotecas/cvWiener2.h"
#include "Bibliotecas/equalization.h"
#include "Bibliotecas/thinning.h"
#include "Bibliotecas/minutiae.h"

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>

int minutiaeNearestSize;

void print(Mat im){
    for (int i = 0; i < im.rows; i++)
    {
        for (int j = 0; j < im.cols; j++)
        {
            cout << (int)im.at<uchar>(i, j) << " ";
        }
        cout << endl;
    }
}

int global_mean(Mat img)
{
    int sum=0;
    for (int i = 0; i < img.rows; i++)
    {
        for (int j = 0; j < img.cols; j++)
        {
            sum += img.at<uchar>(i, j);
        }
    }
    return sum/(img.rows*img.cols);
}

int local_mean(Mat img, int x, int y)
{
    int sum=0;
    for (int i = 0; i < 16 && i < img.rows; i++)
    {
        for (int j = 0; j < 16 && j < img.cols; j++)
        {
            sum += img.at<uchar>(i+x, j+y);
        }
    }
    return sum/16;
}

Mat segmentation(Mat img)
{
    Mat res = img.clone();

    int T = global_mean(img)/2;

    for (int i = 0; i < img.rows; i+=16)
    {
        for (int j = 0; j < img.cols; j+=16)
        {
            if(local_mean(img, i, j) < T)
            {
                for(int k=i;k < i+16 && k < img.rows;++k)
                {
                    for(int l=j;l < j+16 && l < img.cols;++l)
                    {
                        res.at<uchar>(k, l) = 255;
                    }
                }
            }
        }
    }
    return res;
}

int mediana(Mat img, int pos=127)
{
    vector< pair<uchar, int> > vet;

    uchar * res = imgHistogram(img);

    for(int i=0; i < 256;i++){
        vet.push_back(make_pair(res[i],i));
    }
    sort(vet.begin(), vet.end());
    return vet[pos].second;
}

void cvWiener2ADP(Mat srcArr, Mat dstArr, int szWindowX, int szWindowY) {
    IplImage *tmp = new IplImage(srcArr);
    IplImage *tmp2 = cvCreateImage(cvSize(tmp->width, tmp->height), IPL_DEPTH_8U, 1);

    cvWiener2(tmp, tmp2, szWindowX, szWindowY);

    dstArr = Mat(tmp2, true);
}

Mat ce(Mat img)
{
    int quartil = mediana(img);
    Mat aux = img.clone();
    for (int i = 0; i < img.rows; i++)
    {
        for (int j = 0; j < img.cols; j++)
        {
            //cout << (int)img.at<uchar>(i, j) << " ";
            if((int)img.at<uchar>(i, j) < quartil)
            {
                aux.at<uchar>(i, j) = 0;
            }
            else
            {
                aux.at<uchar>(i, j) = 255;
            }
        }
        //cout << endl;
    }
    return aux;
} 

string convertChar(int x){
    switch(x) {
        case 0 :
        return "0";
        break;
        case 1 :
        return "1";
        break;
        case 2 :
        return "2";
        break;
        case 3 :
        return "3";
        break;
        case 4 :
        return "4";
        break;
        case 5 :
        return "5";
        break;
        case 6 :
        return "6";
        break;
        case 7 :
        return "7";
        break;
        case 8 :
        return "8";
        break;
        case 9 :
        return "9";
        break;

    }

}

string intToString(int x, bool zeros){
    if(x == 0){
        if(zeros) return "000";
        else return "0";
    }

    string str = "";
    while(x!=0){
        str = convertChar(x%10) + str;
        x /= 10;
    }

    if(str.size() == 1 && zeros) return "00"+str;
    if(str.size() == 2 && zeros) return "0"+str;
    return str;
}

Mat preProcess(Mat &img){

    Mat res = Mat(img.rows, img.cols, CV_8UC1);

    blur( img, res, Size( 3, 3 ), Point(-1,-1) );
    Mat aux = res.clone();
    bilateralFilter ( aux, res, 5, 5*2, 5/2 );
    aux = res.clone();
    cv::GaussianBlur(aux, res, cv::Size(0, 0), 3);
    cv::addWeighted(aux, 1.5, res, -0.5, 0, res);

    //Filtro de Wiener
    cvWiener2ADP(res, res, 5, 5);

    //Binarizacao e Afinamento
    threshold(res, res, mediana(res), 255, THRESH_BINARY_INV);

    //Esqueletização
    thinning(res);

    /*namedWindow("Preprocess", CV_WINDOW_AUTOSIZE);
    imshow("Preprocess", res);
    waitKey(0);*/
    return res;
}

//distancia entre dois pontos
double distancia(pair<int, int> &a, pair<int, int> &b){

    return sqrt((a.first - b.first)*(a.first - b.first) + (a.second - b.second)*(a.second - b.second));

}

//Retorna as 100 minucias mais próximas
vector<double> nearestMinutiaes(Mat &img){

    vector< pair<int, int> > min = minutiae(img); // Posições das minucias 
    vector<double> dist;

    for(int i = 0; i < min.size()-1; i++){
        for(int j = i+1; j < min.size() ; j++){
            dist.push_back( distancia(min[i], min[j]) );
        }
    }

    sort(dist.begin(), dist.end());

    if(dist.size() > minutiaeNearestSize){
        dist = vector<double> (dist.begin(),dist.begin()+minutiaeNearestSize);
    }

    return dist;
}

int main(int argc, char *argv[]){
    minutiaeNearestSize = 100;
    double threshold = 0.5;

    int imgTraningSize = 5;

    cout << "USUARIO DEDO TX0 TX1 TX2 TX3 TX4 MEDIA\n" << endl;

    for(int user=0;user < 500;user++){
        vector< vector<double> > mat;

        for(int i = 0; i < imgTraningSize; i++){
            string str = "CASIA-FingerprintV5/"+intToString(user, true)+"/L/"+intToString(user, true)+"_L0_"+intToString(i, false)+".bmp";

            Mat img = imread(str, CV_LOAD_IMAGE_GRAYSCALE); //Leitura
            Mat res = preProcess(img); // PreProcessamento, aplicação de filtros e esqueletização
            mat.push_back(nearestMinutiaes(res)); // inclui as distâncias das minucias mais próximas
        }

        for(int ignorar = 0; ignorar < imgTraningSize; ++ignorar){
            printf("%03d L0 ", user);
            double avg = 0;
            for(int j = 0; j < mat.size(); j++){
                int hit = 0;
                if(j != ignorar){
                    for(int k = 0, l = 0; k < mat[j].size() && l < mat[ignorar].size();){
                        double inf = mat[j][k] - threshold;
                        double sup = mat[j][k] + threshold;
                        
                        if(mat[ignorar][l] >= inf && mat[ignorar][l] <= sup){
                            hit++;
                            k++;
                            l++;
                        }
                        else
                        {
                            if(mat[ignorar][l] < inf) l++;
                            else k++;
                        }
                    }
                    avg+=(hit*100)/(double)mat[ignorar].size();
                    printf("%.5lf\t", hit/(double)mat[ignorar].size());
                }
                else{
                    printf("XXX\t");
                }
            }
            printf("%.5lf\n", (avg / 4.0));
        }
        printf("\n");
    }
}

//antiga main 
/*
int main(int argc, char *argv[]) {

    int num_files=5;

    //if (argc <= 1) {
    //	printf("Usage: %s <image>\n", argv[0]);
    //	return 0;
    //}

    Mat training_mat(num_files-1, 328*356, CV_32FC1);
    Mat training_labels(num_files-1, 1, CV_32FC1);

    Mat test_mat(1, 328*356, CV_32FC1);

    for(int i=0;i < num_files;++i){
        //IplImage *tmp = cvLoadImage(argv[1]);
        //IplImage *tmp2 = cvCreateImage(cvSize(tmp->width, tmp->height), IPL_DEPTH_8U, 1);
        //Mat img = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
        cout << "Images/006_L0_"+ intToString(i) +".bmp" << endl;
        Mat img = imread("Images/006_L0_"+ intToString(i) +".bmp", CV_LOAD_IMAGE_GRAYSCALE);
        Mat res = Mat(img.rows, img.cols, CV_8UC1);

        //cvCvtColor(tmp, tmp2, CV_RGB2GRAY);

        //cvNamedWindow("Before");
        //cvShowImage("Before", tmp);
        //namedWindow("Original", CV_WINDOW_AUTOSIZE);
        //imshow("Original", img);

        //Equalizacao

        //res = segmentation(img);
        //namedWindow("Segmentation", CV_WINDOW_AUTOSIZE);
        //imshow("Segmentation", res);
        //waitKey(0);

        blur( img, res, Size( 3, 3 ), Point(-1,-1) );
        Mat aux = res.clone();
        bilateralFilter ( aux, res, 5, 5*2, 5/2 );
        aux = res.clone();
        cv::GaussianBlur(aux, res, cv::Size(0, 0), 3);
        cv::addWeighted(aux, 1.5, res, -0.5, 0, res);

        //Quartil
        //res = ce(res);
        //namedWindow("Equalization", CV_WINDOW_AUTOSIZE);
        //imshow("Equalization", res);

        //Filtro de Wiener
        cvWiener2ADP(res, res, 5, 5);

        //namedWindow("Filter", CV_WINDOW_AUTOSIZE);
        //imshow("Filter", res);

        //Binarizacao e Afinamento
        threshold(res, res, mediana(res), 255, THRESH_BINARY_INV);
        //namedWindow("Binarization", CV_WINDOW_AUTOSIZE);
        //imshow("Binarization", res);

        thinning(res);
        //namedWindow("Thinning", CV_WINDOW_AUTOSIZE);
        //imshow("Thinning", res);

        //string imagename = string(argv[1]);
        //imwrite(imagename+"_thin.bmp", res);

        Mat training_data(1, 328*356, CV_32FC1);
        res = minutiae(res, training_data);
        namedWindow("Minutiae", CV_WINDOW_AUTOSIZE);
        imshow("Minutiae", res);
        //print(res);
        //imwrite("imagem", res);
        //cvNamedWindow("After");
        //cvShowImage("After", tmp2);

        //cvSaveImage("C:/temp/result.png", tmp2);
        //cvWaitKey(-1);

        if(i == 4)
        {
            test_mat = training_data.clone();
        }
        else
        {
            for (int j = 0; j < training_mat.cols; j++) {
                training_mat.at<float>(i, j) = training_data.at<uchar>(0,j);
            }
            training_labels.at<float>(i, 0) = 1;
        }

    }




    CvSVMParams params = CvSVMParams ();
    params.svm_type = CvSVM :: ONE_CLASS ;
    params.kernel_type = CvSVM :: LINEAR ;
    //params.degree = 0; // for poly
    //params.gamma = 20; // for poly / rbf / sigmoid
    //params.coef0 = 0; // for poly / sigmoid
    //params.C = 7; // for CV_SVM_C_SVC , CV_SVM_EPS_SVR and CV_SVM_NU_SVR
    params.nu = 0.5; // for CV_SVM_NU_SVC , CV_SVM_ONE_CLASS , and CV_SVM_NU_SVR
    //params.p = 0.0; // for CV_SVM_EPS_SVR
    //params.class_weights = NULL ; // for CV_SVM_C_SVC
    params.term_crit.type = CV_TERMCRIT_ITER | CV_TERMCRIT_EPS ;
    params.term_crit.max_iter = 1000;
    params.term_crit.epsilon = 1e-6;

    CvSVMParams params;
    params.svm_type    = CvSVM::C_SVC;
    params.kernel_type = CvSVM::LINEAR;
    params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);

    CvSVM svm;
    svm.train(training_mat, training_labels, Mat(), Mat(), params);

    cout << "AQUI" << endl;

    float result;
    result = svm.predict(test_mat);

    waitKey(0);


    //cvReleaseImage(&tmp);
    //cvReleaseImage(&tmp2);

    return 0;

}*/
