#include <opencv2/opencv.hpp>

using namespace std;

// Types
typedef float T;
typedef cv::Vec<T,3> VecT; // 3-vector
typedef cv::Mat_<VecT> ImgT; // 3-channel image
typedef cv::Mat_<T> MatT; // 1-channel container

struct component{
	int area;
	int perimeter;
	double compactness;
	cv::Point centroid;

	void calculate(vector<cv::Point> contour){
		area = cv::contourArea(contour);
		cout << "Area: " << area << endl;
		perimeter = contour.size();
		cout << "Perimeter: " << perimeter << endl;
		compactness = (double) pow((double)perimeter, 2) / (double) area;
		cout << "Compactness: " << compactness << endl;

		int sumx=0, sumy=0;
		for (int i = 0; i < contour.size(); i++) {
			sumx += contour.at(i).x;
			sumy += contour.at(i).y;
		}
		centroid = cv::Point(sumx/contour.size(), sumy/contour.size());
		cout << "Centroid: (" << centroid.x << ", " << centroid.y << ")" << endl;
	}
};

struct componentlist{
	vector<component> list;
	void calculate_all(vector<vector<cv::Point> > contours){
		component tmp;
		for (int i = 0; i < contours.size(); i++) {
			tmp.calculate(contours.at(i));
			list.push_back(tmp);
		}
	}
}componentlist;

string path = "D:/Dropbox/7. Semester/VIS/mini project 2/vis1_pro2_sequences/vis1_pro2_sequences/sequence3/";

string get_file(int img){
	stringstream ss;
	ss << path << img << ".png";
	return ss.str();
}

void find_marker_from_gray_image(string file) {
    // Load image
    cv::Mat org = cv::imread(file, 1);
    cv::Mat img = cv::imread(file, 0);

    // Vector for found circles (Only one is found)
    vector<cv::Vec3f> circles;

    // Blur the image and find the marker
    cv::GaussianBlur(img, img, cv::Size(7,7), 1.5, 1.5);
    cv::HoughCircles( img, circles, CV_HOUGH_GRADIENT, 2, 1000, 200, 100, 45, 50);

    // Paint the marker
    cv::Point center(cvRound(circles[0][0]), cvRound(circles[0][1]));
    int radius = cvRound(circles[0][2]);
    cv::circle( org, center, 3, cv::Scalar(0,255,0), -1, 8, 0 );
    cv::circle( org, center, radius, cv::Scalar(0,0,255), 3, 8, 0 );

    imshow( "circles", org);
    cv::waitKey(10);
}

void animate_one() {
	string file;
    for (int i = 1; i <= 30; i++) {
        file = get_file(i);
        find_marker_from_gray_image(file);
    }
    cv::waitKey();
}

void load_test_image(){
	// Load an image with imread
	string file = get_file(0);
	cout << "file: " << file << endl;
	ImgT src = cv::imread(file);
	src /= 255; // Convert to float region
	if (src.data == NULL){
		std::cout << "Missing image file" << std::endl;
		return;
	}
	cout << "Size: " << src.rows << ", " << src.cols << endl;


	cv::imshow("Test Load", src);
	cv::waitKey();
}

void threshold(const MatT& src, MatT& dst, T low, T high){
	for (int i = 0; i < src.rows; i++) {
		for (int k = 0; k < src.cols; k++){
			if ( src.at<T>(i,k) < high && src.at<T>(i,k) > low){
				dst.at<T>(i,k) = 1;
			}else{
				dst.at<T>(i,k) = 0;
			}
		}
	}
}

void threshold(const ImgT& src, ImgT& dst, int channel, T low, T high){
	for (int i = 0; i < src.rows; i++) {
		for (int k = 0; k < src.cols; k++){
			if ( src.at<VecT>(i,k)[channel] < high && src.at<VecT>(i,k)[channel] > low){
				dst.at<VecT>(i,k)[channel] = 1;
			}else{
				dst.at<VecT>(i,k)[channel] = 0;
			}
		}
	}
}

T max_value(const MatT& src){
	T max = 0;
	for (int i = 0; i < src.rows; i++) {
		for (int k = 0; k < src.cols; k++) {
			if (src.at<T>(i,k) > max){
				max = src.at<T>(i,k);
				cout << "max: " << max << endl;
			}
		}
	}
	return max;
}

void use_bitmask(const MatT& mask, const ImgT& src, ImgT& dst){
	for (int i = 0; i < src.rows; i++) {
		for (int k = 0; k < src.cols; k++){
			if ( mask.at<T>(i,k) != 1){
				dst.at<VecT>(i,k) = VecT(0,0,0);
			}else{
				dst.at<VecT>(i,k) = src.at<VecT>(i,k);
			}
		}
	}
}

vector<vector<cv::Point> > get_valid_contours(vector< vector<cv::Point> >& contours, int minsize){
	vector< vector<cv::Point> > tmp;
	for (int i = 0; i < contours.size(); i++) {
		//cout << contours.at(i).size() << endl;
		if (contours.at(i).size() > minsize){
			tmp.push_back(contours.at(i));
		}
	}
	cout << "Contours before: "<< contours.size() << endl;
	cout << "Contours after: "<< tmp.size() << endl;
	return tmp;
}
void colour_segmentaion(string file){
	ImgT src = cv::imread(file);
	src /= 255; // Convert to float region
	if (src.data == NULL){
		std::cout << "Missing image file: " << file << std::endl;
		return;
	}
	cout << "Size: " << src.rows << ", " << src.cols << endl;

	// convert to HSV
	ImgT hsvimg;
	cv::Mat element;
	cv::cvtColor(src, hsvimg, CV_BGR2HSV, 0);
	// threshold H value
	threshold(hsvimg, hsvimg, 0, 150, 200);
	vector<MatT> hsv_split(3);
	cv::split(hsvimg,hsv_split);
	cv::erode(hsv_split.at(0), hsv_split.at(0), element);
	cv::dilate(hsv_split.at(0),hsv_split.at(0), element);
	use_bitmask(hsv_split.at(0), hsvimg, hsvimg);
	// threshold S
	threshold(hsvimg, hsvimg, 1, 0.25, 0.5);
	cv::split(hsvimg,hsv_split);
	cv::erode(hsv_split.at(1), hsv_split.at(1), element);
	cv::dilate(hsv_split.at(1),hsv_split.at(1), element);
	use_bitmask(hsv_split.at(1), hsvimg, hsvimg);
	// threshold V
	threshold(hsvimg, hsvimg, 2, 0.1, 0.55);
	cv::split(hsvimg,hsv_split);
	cv::erode(hsv_split.at(2), hsv_split.at(2), element);
	cv::dilate(hsv_split.at(2),hsv_split.at(2), element);
	use_bitmask(hsv_split.at(2), hsvimg, hsvimg);

	// Find contours
	cv::Mat binary;
	binary = hsv_split.at(2) * 255;
	binary.convertTo(binary, CV_8U);
	cv::Mat canny_output;
	// Detect edges using canny
	cv::GaussianBlur(binary,binary,cv::Size(7,7),1.5,1.5);
	double thresh = 2.0;
	cv::Canny( binary, canny_output,thresh, thresh*2,3);
	vector< vector<cv::Point> > contours;
	cv::findContours(binary, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
	contours = get_valid_contours(contours, 70);
	/// Draw contours
	cv::Mat drawing = cv::Mat::zeros( canny_output.size(), CV_8UC3 );
	cv::RNG rng(12345);
	for( int i = 0; i< contours.size(); i++ ){
		cv::Scalar color = cv::Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
		cv::drawContours( drawing, contours, i, color, 1);
		cv::Scalar color2 = cv::Scalar( rng.uniform(0, 1), rng.uniform(0,1), rng.uniform(0,1) );
		cv::drawContours( src, contours, i, color, 1);
	}

	// create components
	componentlist.calculate_all(contours);
	int step = 255/componentlist.list.size();
	for (int i = 0; i < componentlist.list.size(); i++) {
		cv::circle(drawing, componentlist.list.at(i).centroid,1,cv::Scalar(0,255,0),1);
		cv::circle(src, componentlist.list.at(i).centroid,1,cv::Scalar(0,step*i,0),1);
	}

	// Show windows
	cv::split(hsvimg,hsv_split);
	cv::normalize(hsv_split.at(0), hsv_split.at(0),360);	// H value 0 - 360 -> 0 - 1

	//cv::cvtColor(hsvimg, src, CV_HSV2BGR);
	//use_bitmask(hsv_split.at(2), src, src);
	//cv::imshow("HSV", hsvimg);
	cv::imshow("Original", src);
	//cv::imshow("Canny", canny_output);
	//cv::imshow("Colour", drawing);
	cv::imshow("H", hsv_split.at(0));
	cv::imshow("S", hsv_split.at(1));
	cv::imshow("V", hsv_split.at(2));
}

void animate(){
	for (int i = 0; i < 30; i++) {
		string file = get_file(i);
		colour_segmentaion(file);
		cv::waitKey(10);
	}
}
int main(int argc, char **argv) {
	//load_test_image();
	//animate_one();
	string file = get_file(30);
	colour_segmentaion(file);
	//animate();
	cv::waitKey();
	return 0;
}
