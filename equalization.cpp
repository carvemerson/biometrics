#include <stdio.h>

#include "equalization.h"
#include <cv.h>

using namespace std;
using namespace cv;

uchar * imgHistogram(const Mat img)
{
	double dumb;
	int i, j;
	double * aux = new double[256];
	uchar * res = new uchar[256];

	for(i=0;i < 256;++i)
	{
		aux[i] = res[i] = 0;
	}

	for(i=0;i < img.rows;++i)
	{
		for(j=0;j < img.cols;++j)
		{
			aux[img.at<uchar>(i, j)] += 1;
		}
	}
	for(i=0;i < 256;++i)
	{
		aux[i] /= (img.rows*img.cols);
	}

	for(i=0;i < 256;++i)
	{
		for(dumb=0, j=0;j <= i;++j)
		{
			dumb += aux[j];
		}
		res[i] = (uchar)(dumb*255);
	}

	return res;
}

Mat imgEnhancementEqualization(const Mat img)
{
	int i, j;
	uchar * h = imgHistogram(img);
	Mat res = Mat(img.rows, img.cols, CV_8UC1);

	for(i=0;i < img.rows;++i)
	{
		for(j=0;j < img.cols;++j)
		{
			res.at<uchar>(i, j) = h[img.at<uchar>(i, j)];
		}
	}

	return res;
}