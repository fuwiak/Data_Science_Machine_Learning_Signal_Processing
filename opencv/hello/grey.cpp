#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
 
int main( int argc, char** argv ) {
  
  cv::Mat image;
  image = cv::imread("cat.jpg" , CV_LOAD_IMAGE_COLOR);
  
  if(! image.data ) {
      std::cout <<  "Could not open or find the image" << std::endl ;
      return -1;
    }

  cv::Mat gray_image;
  cv::cvtColor( image, gray_image, CV_BGR2GRAY );

  cv::imwrite( "Gray_Image.jpg", gray_image );


  cv::imshow( "Gray image", gray_image );
  
  cv::waitKey(0);
  return 0;
}