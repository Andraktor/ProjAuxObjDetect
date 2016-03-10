// CIRAFIDetection.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <vector>
#include <iostream>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\features2d\features2d.hpp>
#include "CIRAFI.hpp"

using namespace cv;
using namespace std;
using namespace CIRAFI;

Mat frame; // matrix to hold frame under analysis
Mat imgHSV; // matrix to hold frame converted to HSV colour space
Mat imgThresh; // matrix to hold thresholded image

vector<vector<Point>> contours;
vector<Vec4i> hierarchy;

// HSV values for red
//int maxH = 179; int minH = 160;
//int maxS = 255; int minS = 0;
//int maxV = 255; int minV = 0;

// HSV values for white
int maxH = 255; int minH = 0;
int maxS = 50; int minS = 0;
int maxV = 255; int minV = 180;

int libLen = 1; // length must be 1 less than total number of images
string libPath = "Template Library\\";
vector<string> strID = { "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"};

int main()
{

	VideoCapture camera(0); // connect to default camera (device 0)
	if (!camera.isOpened()) // if failed to connect to camera, end program
	{
		cout << "Failed to connect to camera. Quitting program." << endl;
		cin.get();
		return -1;
	}

	double camWidth = camera.get(CAP_PROP_FRAME_WIDTH); // get width of images from camera
	double camHeight = camera.get(CAP_PROP_FRAME_HEIGHT); // get height of images from camera

	//Initialise Display Windows
	namedWindow("Camera Input", CV_WINDOW_NORMAL);
	namedWindow("Thresholded Image", CV_WINDOW_NORMAL);

	Mat tImg;
	//Initialise Template Library
	for (int libIndex = 0; libIndex < libLen; libIndex++)
	{
		string pathname = libPath + "IMG_" + to_string(libIndex+1) + ".jpg";
		//Mat tImg = imread(pathname);
		tImg = imread("C:/Users/Infla/OneDrive/Documents/Visual Studio 2015/Projects/CIRAFIDetection/CIRAFIDetection/Template Library/Acrop.jpg");
		if (!tImg.data)
		{
			cout << "Failed to load image " + to_string(libIndex + 1) + " correctly. Quitting program." << endl;
			cin.get();
			return -1;
		}

		resize(tImg, tImg, Size(120, 160));
		cvtColor(tImg, tImg, CV_BGR2HSV);
		inRange(tImg, Scalar(0, 0, 100), Scalar(255, 100, 255), tImg);
	}
	imshow("Template Image", tImg);
	CiratefiData tempA;
	tempA.CountParameter(tImg);
	tempA.Cissq(tImg);
	tempA.Rassq(tImg);

	while (1)
	{
		bool bCheck = camera.read(frame); // read frame from camera
		if (!bCheck)
		{
			cout << "Failed to read frame from camera." << endl;
			break;
		}

		// Resize frame to 640x480
		resize(frame, frame, Size(640, 480));
		
		// Colourspace Conversions
		cvtColor(frame, imgHSV, CV_BGR2HSV); // convert from RGB to HSV colour space
		inRange(imgHSV, Scalar(minH, minS, minV), Scalar(maxH, maxS, maxV), imgThresh); // threshold image based on set parameters

		// Morphological Operations
		// Opening
		erode(imgThresh, imgThresh, getStructuringElement(MORPH_ELLIPSE, Size(5, 5))); // apply erosion to image
		dilate(imgThresh, imgThresh, getStructuringElement(MORPH_ELLIPSE, Size(5, 5))); // apply dilation to image

		// Closing
		dilate(imgThresh, imgThresh, getStructuringElement(MORPH_ELLIPSE, Size(5, 5))); // apply dilation to image
		erode(imgThresh, imgThresh, getStructuringElement(MORPH_ELLIPSE, Size(5, 5))); // apply erosion to image
		
		// Find seperate object bounding boxes in image
		Mat contImg = imgThresh.clone();
		cv::findContours(contImg, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
		for (int i = 0; i < contours.size(); i++)
		{
			Rect bBox = boundingRect(contours[i]);
			if (bBox.width < 80 || bBox.height < 80) continue;
			Mat roi = imgThresh(bBox);
			imshow("ROI", roi);
			tempA.Cisssa(roi);
			tempA.Cifi(roi);
			if (!tempA._cis.empty())
			{
				tempA.Rafi(roi);
				tempA.Tefi(roi, tImg);
			}
		}

		// Evaluate results of CIRAFI analysis and display to window
		if (!tempA._cis.empty())
		{
			int ct = tempA._cis.size() - 1;
			double cis = tempA._cis[ct].GetCoefficient();
			cout << "CIFI: " + to_string(cis) << endl << endl;

			if (!tempA._ras.empty())
			{
				int rt = tempA._ras.size() - 1;
				double ras = tempA._ras[rt].GetCoefficient();
				double tot = cis*ras;
				cout << "RAFI: " + to_string(ras) << endl << endl;
				cout << "TOTAL: " + to_string(tot) << endl << endl;
			}
		}

		imshow("Camera Input", frame); // show original frame
		resizeWindow("Camera Input", 640, 480);
		imshow("Thresholded Image", imgThresh); // show thresholded image
		resizeWindow("Thresholded Image", 640, 480);

		if (waitKey(30) == 27)
		{
			cout << "Process ended by user" << endl;
			return -1;
		}
	}
    return 0;
}

