#include "Bibliotecas/minutiae.h"

bool MaskMatch(int x, int y, Mat &im, Mat &mask){
    int k, l, i, j;
    bool res = true;
    for (i=x-1, k = 0; k < mask.rows && res; i++, k++)
    {
        for (j = y - 1, l = 0; l < mask.cols && res; j++, l++)
        {
            res = res && (im.at<uchar>(i, j) == mask.at<uchar>(k, l));

        }
    } 
    if(res)       
        cout <<  x<< " " << y <<endl;                  
    return res;
}

bool EndLine(int x, int y, Mat &im){
    int res = 0;

    res +=  (int)(im.at<uchar>(x-1, y-1) == 255);
    res +=  (int)(im.at<uchar>(x-1, y) == 255);
    res +=  (int)(im.at<uchar>(x-1, y+1) == 255);
    res +=  (int)(im.at<uchar>(x, y-1) == 255);
    res +=  (int)(im.at<uchar>(x, y+1) == 255);
    res +=  (int)(im.at<uchar>(x+1, y-1) == 255);
    res +=  (int)(im.at<uchar>(x+1, y) == 255);
    res +=  (int)(im.at<uchar>(x+1, y+1) == 255);


    if(res == 1)return true;

    return false;

}

bool Bifurcation(int x, int y, Mat &im){
    int res = 0;

    res +=  (int)(im.at<uchar>(x-1, y-1) == 255);
    res +=  (int)(im.at<uchar>(x-1, y) == 255);
    res +=  (int)(im.at<uchar>(x-1, y+1) == 255);
    res +=  (int)(im.at<uchar>(x, y-1) == 255);
    res +=  (int)(im.at<uchar>(x, y+1) == 255);
    res +=  (int)(im.at<uchar>(x+1, y-1) == 255);
    res +=  (int)(im.at<uchar>(x+1, y) == 255);
    res +=  (int)(im.at<uchar>(x+1, y+1) == 255);


    if(res == 3)return true;

    return false;
}

// Retorna as possições das  minucias
vector < pair<int, int> > minutiae(Mat &img){
    
    Mat thr = img.clone();
    vector < pair<int, int> > res;

    for (int i = 1; i < thr.rows-1; i++){
        for (int j = 1; j < thr.cols-1; j++){
            if(img.at<uchar>(i, j) == 255){
                if(EndLine(i, j, img)){
                    circle(thr, Point(j, i), 10, Scalar(255,255,255), 1);
                    res.push_back(make_pair(i,j));
                }
            }
        }
    }
    
   /* namedWindow("Minutiae", CV_WINDOW_AUTOSIZE);
    imshow("Minutiae", thr);
    waitKey(0);*/
    
    return res;
}

/* Minutiae Antiga
Mat minutiae(Mat &im, Mat &dst){

    Mat thr = im.clone();
    Mat res = Mat(1, im.rows*im.cols, CV_8UC1);


    //Mat mask=(Mat_<uchar>(3,3)<<  0,  0, 0,\
    //                              255, 255, 0,\
    //                              0, 0,  0);  


    for (int i = 1; i < thr.rows-1; i++)
    {
        for (int j = 1; j < thr.cols-1; j++)
        {
            if(im.at<uchar>(i, j) == 255){
                /* if(Bifurcation(i, j, im)){
		       		circle(thr, Point(j, i), 10, Scalar(255,255,255), 1);
		       		res.at<uchar>(0, j+i*j) = 0;
		       }
		       else
                if(EndLine(i, j, im)){
                    circle(thr, Point(j, i), 10, Scalar(255,255,255), 1);
                    res.at<uchar>(0, j+i*j) = 255;
                }
                else
                {
                    res.at<uchar>(0, j+i*j) = 127;
                }
            }
        }
    }                           

    dst = res.clone();

    return thr;
}*/
