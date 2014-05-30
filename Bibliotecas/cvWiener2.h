#include <cv.h>

//
// cvWiener2  - Applies Wiener filtering on a 2D array of data
//   Args:
//      srcArr     -  source array to filter
//      dstArr     -  destination array to write filtered result to
//      szWindowX  -  [OPTIONAL] length of window in x dimension (default: 3)
//      szWindowY  -  [OPTIONAL] length of window in y dimension (default: 3)
//
void cvWiener2( 
					const void* srcArr, 
					void* dstArr,
					int szWindowX = 3, 
					int szWindowY = 3
		  );