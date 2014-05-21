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
 
 Mat minutiae(Mat &im){
 
 	Mat thr = im.clone();
 	

    Mat mask=(Mat_<uchar>(3,3)<<  0,  0, 0,\
                                  255, 255, 0,\
                                  0, 0,  0);  
                                   
                                
    for (int i = 1; i < thr.rows-1; i++)
    {
        for (int j = 1; j < thr.cols-1; j++)
        {
		    if(im.at<uchar>(i, j) == 255){
		       if(MaskMatch(i, j, im, mask)){
		       		circle(thr, Point(j, i), 10, Scalar(255,255,255), 1);
		       }
		       if(EndLine(i, j, im)){
		       		circle(thr, Point(j, i), 10, Scalar(255,255,255), 1);
		       }
		    }
        }
    }                           
   
   return thr;
}
