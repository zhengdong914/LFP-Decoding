//
// Created by xiaozd on 12/24/18.
//

#ifndef LFP_DECODING_ALGORITHM_H
#define LFP_DECODING_ALGORITHM_H

#include <iostream>
#include <vector>
#include <string>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/SparseCore>

class Modeldecoder
{
protected:
    bool init;
    // data param
    float binSize; // in second
    int units;
    int seq_l; // train seq length, will be set during training
    // model params
    float a;
    float q;
    Eigen::VectorXf c;
    Eigen::VectorXf d;
    Eigen::VectorXf xsm;
    Eigen::VectorXf Vsm;
    // last state, updated after training and every decoding call
    float x0;
    float q0;
    // baseline
    float meanBase;
    float stdBase;
public:
    // constructor
    Modeldecoder() :
            init(false),
            binSize(0.05)
    {}


    // sets,gets and inline methods
    bool isInit() const {return init;}
    void setInit() { init = true; }
    void setUnits(int x) { units = x; }
    float get_binSize() const { return binSize; }
    float getA() const { return a; }
    void setA(float x) { a = x; }
    float getX0() const { return x0; }
    void setX0(float x) { x0 = x; }
    float getQ0() const { return q0; }
    void setQ0(float x) { q0 = x; }
    float getQ() const { return q; }
    void setQ(float x) { q = x; }
    Eigen::VectorXf getC() const  { return c; }
    void setC(Eigen::VectorXf x) { c = x; }
    Eigen::VectorXf getD() const { return d; }
    void setD(Eigen::VectorXf x) { d = x; }
    Eigen::VectorXf getXsm() const { return xsm; }
    Eigen::VectorXf getVsm() const { return Vsm; }
    float get_meanBase() const { return meanBase; }
    void set_meanBase(float x) { meanBase = x; }
    float get_stdBase() const { return stdBase; }
    void set_stdBase(float x) { stdBase = x; }

    float getZscore() const
    {
        if (abs(stdBase) < 1e-6)
        {
            return 0;
        }
        else
        {
            return (x0 - meanBase) / stdBase;
        }
    }

    float getZscore(const float x_) const
    {
        if (abs(stdBase) < 1e-6)
        {
            return 0;
        }
        else
        {
            return (x_ - meanBase) / stdBase;
        }
    }

    float getConf(const float nStd) const
    {
        if (abs(stdBase) < 1e-6 || !isInit())
        {
            return 0;
        }
        else
        {
            return sqrt(q0)*nStd / stdBase;
        }
    }

    float getConf(const float q_,const float nStd) const
    {
        if (abs(stdBase) < 1e-6 || !isInit())
        {
            return 0;
        }
        else
        {
            return sqrt(q_)*nStd / stdBase;
        }
    }

    void set_binSize(float binSize_){ binSize = binSize_; }

    // methods
    virtual void train(const Eigen::MatrixXf& data) = 0;
    virtual float decode(const Eigen::VectorXf& data) = 0;
    virtual void updateBaseline(const Eigen::VectorXf& data);
    virtual ~Modeldecoder(){}
};

enum TLDSTYPE{ LOGTLDS, SQROOTLDS };

class TLDSdecoder :public Modeldecoder
{
private:
    // em param
    int emMaxIter;
    int faMaxIter;
    float emTol;
    float faTol;
    // tlds type
    TLDSTYPE tldsType;
    Eigen::MatrixXf tSeq;
    // model param
    Eigen::VectorXf r; //observation noise covariance
    Eigen::VectorXf ll; //log likelihood curve
    void faInit(const Eigen::MatrixXf & data);
public:
    // constructor
    TLDSdecoder() :
            emMaxIter(300), faMaxIter(100), emTol(1e-6), faTol(0.001),
            tldsType(LOGTLDS)
    {}


    // sets,gets and inline methods
    Eigen::VectorXf getR() const{ return r; }
    void setR(Eigen::VectorXf x) { r = x; }
    Eigen::VectorXf getLL() const { return ll; }
    void setLL(Eigen::VectorXf x) { ll = x; }
    int getEmMaxIter() const { return emMaxIter; }
    int getFaMaxIter() const { return faMaxIter; }
    float getEmTol() const { return emTol; }
    float getFaTol() const { return faTol; }
    TLDSTYPE getTldsType() const { return tldsType; }

    void setEmMaxIter(const int emMaxIter_){ emMaxIter = emMaxIter_; }
    void setFaMaxIter(const int faMaxIter_){ faMaxIter = faMaxIter_; }
    void setEmTol(const float emTol_){ emTol = emTol_; }
    void setFaTol(const float faTol_){ faTol = faTol_; }
    void setTldsType(const TLDSTYPE tldsType_){ tldsType = tldsType_; }

    virtual void train(const Eigen::MatrixXf& data);
    virtual float decode(const Eigen::VectorXf& data);
    virtual ~TLDSdecoder(){}
};

#endif //LFP_DECODING_ALGORITHM_H
