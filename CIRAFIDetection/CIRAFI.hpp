#pragma once
#include "stdafx.h"
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\features2d\features2d.hpp>
#include <iostream>
#include <numeric>
#include <opencv2\imgproc\imgproc.hpp>

using namespace cv;
using namespace std;

#ifndef _CIRAFI_H_
#define _CIRAFI_H_

#define PI 3.14159265359

#define initScale 0.5
#define finScale 2.5
#define numScale 9

#define scaleThresh 0.8
#define angThresh 0.7

namespace CIRAFI
{
	struct LetterData
	{
		LetterData() : coef(-1), letter('-') {};
		LetterData(char let, double scr)
		{ 
			letter = let;
			coef = scr;
		}
		char letter;
		double coef;
	};

	class ObjectData
	{
	public:
		ObjectData(int tempRad) : _circleNum(16), _initialRadius(0), _scaleNum(numScale), _initialScale(initScale), _finalScale(finScale)
		{
			if (_scaleNum>1) _passoesc = exp(log(_finalScale / _initialScale) / _scaleNum); else _passoesc = 1.0;
			SetTemplateRadius(tempRad);
		}

		void ObjectAnalysis(cv::Mat& sourceImage);
		std::vector<double> Get(void) { return _ca; }

	private:

		double CircularSample(cv::Mat& image, int y, int x, int radius);
		void SetTemplateRadius(int templateRadius);
		template <class T>
		T clip(const T val, const T lower, const T upper) { return std::max(lower, std::min(val, upper)); }
		double scale(double s) { return _initialScale*pow(_passoesc, s); }

		int _circleNum;
		double _initialRadius;
		double _finalRadius;
		double _circleDistance;
		int _scaleNum;
		double _initialScale;
		double _finalScale;
		double _passoesc;
		int _templateRadius;
		std::vector<double> _ca;

	};

	class CorrData
	{
	public:
		CorrData(int row, int col, int scale, int angle, double coefficient) :_row(row), _col(col), _scale(scale), _angle(angle), _coefficient(coefficient) {}
		double GetCoefficient() { return _coefficient; }
		int GetScale() { return _scale; }
		int GetAngle() { return _angle; }
		int GetRow() { return _row; }
		int GetCol() { return _col; }
	private:
		double _coefficient;
		int _scale;
		int _angle;
		int _row;
		int _col;
	};

	class CIRAFIData
	{
	public:
		CIRAFIData() :_scaleNum(numScale), _initialScale(initScale), _finalScale(finScale), _angleNum(36), _scaleThreshold(scaleThresh), _angleThreshold(angThresh), _nccThreshold(0.9)
			, _isMatchNegative(false), _circleNum(16), _initialRadius(0), _finalRadius(-1), _tefiTolerance(1), maxCis(-1,-1,-1,-1,0), maxRas(-1, -1, -1, -1, 0), maxTes(-1, -1, -1, -1, 0), _letter('-') {}

		CIRAFIData(cv::Mat& templateImage, char letter) : _scaleNum(numScale), _initialScale(initScale), _finalScale(finScale), _angleNum(36), _scaleThreshold(scaleThresh), _angleThreshold(angThresh), _nccThreshold(0.9)
			, _isMatchNegative(false), _circleNum(16), _initialRadius(0), _finalRadius(-1), _tefiTolerance(1), maxCis(-1, -1, -1, -1, 0), maxRas(-1, -1, -1, -1, 0), maxTes(-1, -1, -1, -1, 0), _letter(letter)
		{
			TemplateSample(templateImage);
		}

		void CountParameter(cv::Mat& templateImage);
		double scale(double s) { return _initialScale*pow(_passoesc, s); }
		void SetScaleNum(int scaleNum) { _scaleNum = scaleNum; }
		void SetInitialScale(double initialScale) { _initialScale = initialScale; }
		void SetFinalScale(double finalScale) { _finalScale = finalScale; }
		void SetAngleNum(int angleNum) { _angleNum = angleNum; }
		void SetScaleThreshold(double scaleThreshold) { _scaleThreshold = scaleThreshold; }
		void SetAngleThreshold(double angleThreshold) { _angleThreshold = angleThreshold; }
		void SetNccThreshold(double nccThreshold) { _nccThreshold = nccThreshold; }
		void SetMatchNegative(bool isMatchNegative) { _isMatchNegative = isMatchNegative; }
		void SetCircleNum(int circleNum) { _circleNum = circleNum; }
		void SetInitialRadius(double initialRadius) { _initialRadius = initialRadius; }
		void SetTefiTolerance(int tefiTolerance) { _tefiTolerance = tefiTolerance; }

		template <class T>
		T clip(const T val, const T lower, const T upper) { return std::max(lower, std::min(val, upper)); }
		double CircularSample(cv::Mat& image, int y, int x, int radius);
		void Cissq(cv::Mat& templateImage);
		void Cifi(cv::Mat& sourceImage, std::vector<double> ca);

		double RadialSample(cv::Mat& image, int centerY, int centerX, double angle, double radius);
		void Rassq(cv::Mat& templateImage);
		void Rafi(cv::Mat& sourceImage);

		void CIRAFIData::Tefi(cv::Mat& sourceImage, cv::Mat& templateImage);
		cv::Mat DrawTefiResult(cv::Mat& sourceImage, double sampleRatio = 1);

		void TemplateSample(cv::Mat& templateImage);
		void ObjectCompare(cv::Mat& sourceImage, std::vector<double> ca);
		double CalculateCoef(void);
		void ResetCoefficients(void);
		
		char GetTempLetter(void) { return _letter; }

		std::vector<CorrData> _cis;
		std::vector<CorrData> _ras;
		std::vector<CorrData> _tes;
		CorrData maxCis;
		CorrData maxRas;
		CorrData maxTes;

	private:
		int _scaleNum;
		double _initialScale;
		double _finalScale;
		int _angleNum;
		double _scaleThreshold;
		double _angleThreshold;
		double _nccThreshold;
		bool _isMatchNegative;
		int _circleNum;
		double _initialRadius;
		double _finalRadius;
		int _tefiTolerance;
		double _circleDistance;
		double _passoesc;
		double _angleDegree;
		double _angleRadian;
		double _templateRadius;
		std::vector<double> _cq;
		vector<vector<double>> _cqi;
		vector<double> _cqi2;
		std::vector<double> _rq;

		char _letter;
	};

	inline double round(double val, int precision = 0)
	{
		double mul = pow(10, (double)precision);
		val *= mul;
		val = (val<0.0) ? ceil(val - 0.5) : floor(val + 0.5);
		return val / mul;
	}
}


#endif