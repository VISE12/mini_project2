#include <opencv2/opencv.hpp>

using namespace std;

// Types
typedef float T;
typedef cv::Vec<T,3> VecT; // 3-vector
typedef cv::Mat_<VecT> ImgT; // 3-channel image
typedef cv::Mat_<T> MatT; // 1-channel container

struct parameters{
	int file_no;
	double H_low, H_high;
	double S_low, S_high;
	double V_low, V_high;
	int Hv_low, Hv_high;
	int Sv_low, Sv_high;
	int Vv_low, Vv_high;
}parameters;

struct component{
	int area;
	int perimeter;
	double compactness;
	cv::Point centroid;

	void calculate(vector<cv::Point> contour){
		area = cv::contourArea(contour);
		perimeter = contour.size();
		compactness = (double) pow((double)perimeter, 2) / (double) area;

		int sumx=0, sumy=0;
		for (int i = 0; i < contour.size(); i++) {
			sumx += contour.at(i).x;
			sumy += contour.at(i).y;
		}
		centroid = cv::Point(sumx/contour.size(), sumy/contour.size());

//		cout << "Compactness: " << compactness << endl;
//		cout << "Perimeter: " << perimeter << endl;
//		cout << "Area: " << area << endl;
//		cout << "Centroid: (" << centroid.x << ", " << centroid.y << ")" << endl;
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

string path = "D:/Dropbox/7. Semester/VIS/mini project 2/vis1_pro2_sequences/vis1_pro2_sequences/sequence1/";
string winname = "output";

string get_file(int img){
	stringstream ss;
	ss << path << img << ".png";
	return ss.str();
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
	//cout << "Contours before: "<< contours.size() << endl;
	//cout << "Contours after: "<< tmp.size() << endl;
	return tmp;
}
void colour_segmentaion(string file){
	ImgT src = cv::imread(file);
	src /= 255; // Convert to float region
	if (src.data == NULL){
		std::cout << "Missing image file: " << file << std::endl;
		return;
	}
	cout << "file: " << file << endl;
	cout << "Size: " << src.rows << ", " << src.cols << endl;

	// convert to HSV
	ImgT hsvimg;
	cv::Mat element;
	cv::cvtColor(src, hsvimg, CV_BGR2HSV, 0);
	// threshold H value
	//threshold(hsvimg, hsvimg, 0, 150, 200);
	threshold(hsvimg, hsvimg, 0, parameters.H_low, parameters.H_high);
	vector<MatT> hsv_split(3);
	cv::split(hsvimg,hsv_split);
	cv::erode(hsv_split.at(0), hsv_split.at(0), element);
	cv::dilate(hsv_split.at(0),hsv_split.at(0), element);
	use_bitmask(hsv_split.at(0), hsvimg, hsvimg);
	// threshold S
	//threshold(hsvimg, hsvimg, 1, 0.25, 0.5);
	threshold(hsvimg, hsvimg, 1, parameters.S_low, parameters.S_high);
	cv::split(hsvimg,hsv_split);
	cv::erode(hsv_split.at(1), hsv_split.at(1), element);
	cv::dilate(hsv_split.at(1),hsv_split.at(1), element);
	use_bitmask(hsv_split.at(1), hsvimg, hsvimg);
	// threshold V
	//threshold(hsvimg, hsvimg, 2, 0.1, 0.55);
	threshold(hsvimg, hsvimg, 2, parameters.V_low, parameters.V_high);
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

	cv::resize(src, src, cv::Size(0,0), 0.5,0.5);	// Resize the image to fit in stupid notebook window
	cv::imshow(winname, src);
	//cv::imshow("Canny", canny_output);
	//cv::imshow("Colour", drawing);
	//cv::imshow("H", hsv_split.at(0));
	//cv::imshow("S", hsv_split.at(1));
	//cv::imshow("V", hsv_split.at(2));
}

void animate(){
	string file;
	for (int i = 0; i <= 30; i++) {
		file = get_file(i);
		colour_segmentaion(file);
		int c = cvWaitKey(20);
		if((char)c==27){
			break;
		}
	}
}

void my_grayscale(const ImgT& src, cv::Mat& dst){
	dst = MatT(src.size());
	for (int i = 0; i < src.rows; i++) {
		for (int k = 0; k < src.cols; k++) {
			dst.at<T>(i,k) = (src.at<VecT>(i,k)[0]*255/360) * src.at<VecT>(i,k)[1]*src.at<VecT>(i,k)[2];
			//sqrt(pow((double) src.at<T>(i,k)[0],2) + pow((double) src.at<T>(i,k)[1],2)+ pow((double) src.at<T>(i,k)[2],2));
		}
	}
}

void feature_detection(string file){
	ImgT src = cv::imread(file);
	src /= 255; // Convert to float region
	if (src.data == NULL){
		std::cout << "Missing image file: " << file << std::endl;
		return;
	}
	cout << "file: " << file << endl;
	cout << "Size: " << src.rows << ", " << src.cols << endl;

	cv::Mat binary, gray;
	ImgT hsv;
	cv::cvtColor(src, hsv, CV_BGR2HSV);
	my_grayscale(hsv, gray);
	//binary.convertTo(binary, CV_8U);
	gray.convertTo(gray, CV_8U);
	cv::Mat canny_output;
	// Detect edges using canny
	//cv::GaussianBlur(binary,binary,cv::Size(7,7),1.5,1.5);
	cv::GaussianBlur(gray,gray,cv::Size(7,7),1.5,1.5);
	double thresh = 20;
	cv::Canny( gray, canny_output, thresh, thresh*3,3);
	// Try the harris corners function
//	MatT gray;
//	cv::cvtColor(src, gray, CV_BGR2GRAY);
//	MatT corners = cv::Mat::zeros( src.size(), CV_32FC1 );
//	corners.convertTo(corners,CV_32FC1);
//	cv::cornerHarris(canny_output, corners, 2, 3, 0.04);

	cv::resize(src, src, cv::Size(0,0), 0.5,0.5);	// Resize the image to fit in stupid notebook window
	//cv::resize(corners, corners, cv::Size(0,0), 0.5,0.5);	// Resize the image to fit in stupid notebook window
	cv::imshow(winname, src);
	cv::imshow("Gray", gray);
	cv::imshow("Canny", canny_output);
	//cv::imshow("Corner", corners);
}
void callback(int, void*){
	parameters.H_low = (double) parameters.Hv_low;
	parameters.H_high = (double) parameters.Hv_high;
	parameters.S_low = (double) parameters.Sv_low / (double) 1000;
	parameters.S_high = (double) parameters.Sv_high / (double) 1000;
	parameters.V_low = (double) parameters.Vv_low / (double) 1000;
	parameters.V_high = (double) parameters.Vv_high / (double) 1000;
	//cout << "H low: " << parameters.H_low << endl;
	string file = get_file(parameters.file_no);
	colour_segmentaion(file);
}

void animate_parameters(){
	parameters.file_no = 30;
	string trackers = "trackers";
	cv::namedWindow(winname, CV_WINDOW_AUTOSIZE);
	cv::namedWindow(trackers, CV_WINDOW_AUTOSIZE);
	cv::Mat empty = cv::Mat::zeros(cv::Size(400,300),CV_32F);
	cv::createTrackbar("File", trackers, &parameters.file_no, 30, callback);
	cv::createTrackbar("H low", trackers, &parameters.Hv_low, 360, callback);
	cv::createTrackbar("H high", trackers, &parameters.Hv_high, 360, callback);
	cv::createTrackbar("S low", trackers, &parameters.Sv_low, 1000, callback);
	cv::createTrackbar("S high", trackers, &parameters.Sv_high, 1000, callback);
	cv::createTrackbar("V low", trackers, &parameters.Vv_low, 1000, callback);
	cv::createTrackbar("V high", trackers, &parameters.Vv_high, 1000, callback);
	cv::imshow(trackers,empty);
	callback(0,0);
	cv::waitKey();
}

void animate_feature(){
	string file;
	for (int i = 0; i <= 30; i++) {
		file = get_file(i);
		feature_detection(file);
		int c = cv::waitKey(20);
		if (c == 27){
			break;
		}
	}
	cv::waitKey();
}

int main(int argc, char **argv) {
	parameters.Hv_high = 200;
	parameters.Hv_low = 146;
	parameters.Sv_high = 650;
	parameters.Sv_low = 150;
	parameters.Vv_high = 550;
	parameters.Vv_low = 100;

	callback(0,0);
	//load_test_image();
	//animate_one();
	//string file = get_file(30);
	//colour_segmentaion(file);
	//animate_parameters();	// Run this to set your desired parameters - Press esc or close window when done
	//animate();				// This is the free running method

	animate_feature();
	cv::waitKey();
	return 0;
}
