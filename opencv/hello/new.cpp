


#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
 
int main( int argc, char** argv ) {
  
  cv::Mat image;
  image = cv::imread("cat.jpg" , CV_LOAD_IMAGE_COLOR);
  
  cv::Mat new_image = cv::Mat::zeros( image.size(), image.type() );

  cv::imshow( "new_image", new_image );
  

  cv::waitKey(0);
  return 0;
}