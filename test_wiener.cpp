#include <stdio.h>
#include <algorithm>

#include "cvWiener2.h"
#include "equalization.h"
#include <opencv2/opencv.hpp>

int mediana(Mat img)
{
	uchar * res = imgHistogram(img);
	sort(res, res+256);
	return res[127];
}

void cvWiener2ADP(Mat srcArr, Mat dstArr, int szWindowX, int szWindowY) {
	IplImage *tmp = new IplImage(srcArr);
	IplImage *tmp2 = cvCreateImage(cvSize(tmp->width, tmp->height), IPL_DEPTH_8U, 1);
	
	cvWiener2(tmp, tmp2, szWindowX, szWindowY);

	dstArr = Mat(tmp2, true);
}

int main(int argc, char *argv[]) {

	if (argc <= 1) {
		printf("Usage: %s <image>\n", argv[0]);
		return 0;
	}


	//IplImage *tmp = cvLoadImage(argv[1]);
	//IplImage *tmp2 = cvCreateImage(cvSize(tmp->width, tmp->height), IPL_DEPTH_8U, 1);
	Mat img = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
	Mat res = Mat(img.rows, img.cols, CV_8UC1);

	//cvCvtColor(tmp, tmp2, CV_RGB2GRAY);

	//cvNamedWindow("Before");
	//cvShowImage("Before", tmp);
	namedWindow("Original", CV_WINDOW_AUTOSIZE);
	imshow("Original", img);

	//Equalizacao
	res = imgEnhancementEqualization(img);
	
	namedWindow("Equalization", CV_WINDOW_AUTOSIZE);
	imshow("Equalization", res);

	//Filtro de Wiener
	cvWiener2ADP(res, res, 5, 5);
	
	namedWindow("Filter", CV_WINDOW_AUTOSIZE);
	imshow("Filter", res);

	//Binarizacao e Afinamento
	threshold(res, res, mediana(res), 255, THRESH_BINARY);

	namedWindow("Binarization", CV_WINDOW_AUTOSIZE);
	imshow("Binarization", res);
	
	//cvNamedWindow("After");
	//cvShowImage("After", tmp2);	

	//cvSaveImage("C:/temp/result.png", tmp2);
	//cvWaitKey(-1);
	waitKey(0);


	//cvReleaseImage(&tmp);
	//cvReleaseImage(&tmp2);

	return 0;

}
