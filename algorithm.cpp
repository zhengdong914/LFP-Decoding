//
// Created by xiaozd on 12/24/18.
//

#include "algorithm.h"
#include "LBFGS/LBFGS_hsl.h"
#include <cmath>
#define fix_d

void Modeldecoder::updateBaseline(const Eigen::VectorXf& data)
{
    if (data.rows()==0)
    {
        std::cout << "empty data seq"<<std::endl;
        return;
    }
    meanBase = data.mean();
    stdBase = sqrt(data.array().pow(2).mean() - meanBase*meanBase);
}

void TLDSdecoder::faInit(const Eigen::MatrixXf & data)
{
    Eigen::MatrixXf XX(seq_l, units);
    Eigen::MatrixXf cX(units, units);
    Eigen::VectorXf L(units);
    Eigen::MatrixXf epsi(seq_l, units);

    epsi = (Eigen::MatrixXf::Random(seq_l, units)*0.1).array().abs();

    if (tldsType == LOGTLDS)
    {
        // +0.1 to avoid the all-zero data case
        //tSeq = ((data + epsi).array()+1).log();
        tSeq = (data.array() + 1).log();
    }
    else
    {
        // +0.1 to avoid the all-zero data case
        //tSeq = (data + epsi).cwiseSqrt();
        tSeq = data.cwiseSqrt();
    }


    auto tmp = tSeq;

    d = tSeq.colwise().mean();
    tSeq = tSeq.rowwise() - d.transpose();
    XX = tSeq.transpose()*tSeq;
    cX = XX.array() / (seq_l - 1);
    XX = XX.array() / seq_l;

    auto tt  = cX.determinant();

    float scale = pow(cX.determinant(), 1.0 / units);
    //L = Eigen::VectorXf::Random(units)*sqrt(std::fmax(scale,1));
    L = Eigen::VectorXf::Random(units)*sqrt(scale);
    /*
        if (units == 6) L << -0.2290, 1.2478, 0.1201, -1.2045, -1.5084, 0.8386;
        if (units == 7) L << -1.6999, -1.6525, -0.4268, -2.0909, -0.7219, 0.5004, 2.2210;
        if (units == 8) L << 0.3189, 0.4629, -1.6422, -0.3616, 0.8706, -0.5206, 0.0686, 0.5539;
        if (units == 10) L << 0.2216, -0.1376, 0.1761, -0.3911, -0.4380, -2.0001, -0.3125, 0.3393, 0.3206, 0.7681;
        if (units == 12) L << -1.1817, 0.4817, 0.1661, 1.3291, 1.4354, -0.7408, 1.5254, -0.8764, 1.1599, -1.2547, -1.8879, -0.5333;
        if (units == 14) L << 0.3675, -0.0104, -0.5901, 1.2398, -0.1081, -1.4640, -0.0675, 1.3366, 0.2676, -0.2184, -0.9733, -0.8022, 0.2143, 1.7681;
        L = L*sqrt(scale);
        */

    Eigen::VectorXf Ph(units);
    Eigen::VectorXf LP(units);
    Eigen::VectorXf tmpL(units);
    Eigen::MatrixXf MM(units, units);
    Eigen::VectorXf beta(units);
    Eigen::MatrixXf XM(seq_l, units);
    Eigen::VectorXf EZ(seq_l);
    Eigen::VectorXf H(seq_l);

    Ph = cX.diagonal();
    float con = -units / 2.0 * log(2 * EIGEN_PI);
    float dM, EZZ;
    float lik = 0;
    float oldlik = 0;
    float likbase;
    for (int i = 0; i < faMaxIter; i++)
    {
        //E step
        LP = L.cwiseQuotient(Ph);
        MM = Ph.cwiseInverse().asDiagonal();
        MM = MM - LP*(1.0 / (L.dot(LP) + 1))*LP.transpose();
        dM = sqrt(MM.determinant());
        beta = L.transpose()*MM;
        XM = tSeq*MM;
        EZ = XM*L;
        EZZ = 1 - beta.dot(L) + (beta.transpose()*XX*beta);

        //compute log likelihood
        oldlik = lik;
        H = -0.5*(XM.cwiseProduct(tSeq)).rowwise().sum();
        H = H.array() + con + log(dM);
        lik = H.sum();

        //M step
        L = tSeq.transpose()*EZ*(1.0 / (EZZ*seq_l));
        Ph = (XX - L*EZ.transpose()*tSeq*(1.0 / seq_l)).diagonal();

        if (i <= 2)
        {
            likbase = lik;
        }
        else if (lik < oldlik)
        {
            std::cout << "init violation" << std::endl;
        }
        else if (lik - likbase < (1 + faTol)*(oldlik - likbase) || std::isinf(lik))
        {
            break;
        }
        std::cout << "TLDS fa iter="<<i<<std::endl;
    }


    Eigen::MatrixXf Phi(units, units);
    Eigen::VectorXf temp1(units);
    Eigen::MatrixXf temp2(units, units);
    Eigen::VectorXf t1(seq_l - 1);
    Eigen::VectorXf t2(seq_l - 1);
    c = L;
    r = Ph;
    Phi = r.cwiseInverse().asDiagonal();
    temp1 = Phi*L;
    temp2 = Phi - temp1*(1.0 / (1 + L.dot(temp1)))*temp1.transpose();
    temp1 = tSeq*temp2*L;
    x0 = temp1.mean();

    t1 = temp1.head(seq_l - 1);
    t2 = temp1.tail(seq_l - 1);
    temp1 = temp1.array() - x0;
    q = temp1.dot(temp1)*(1.0 / (seq_l - 1));
    q0 = q;
    a = t1.dot(t2) / (t1.dot(t1) + q);
}

void TLDSdecoder::train(const Eigen::MatrixXf& data)
{
    units = data.cols();
    seq_l = data.rows();
    std::cout <<"#feature = "<< units << std::endl;
    std::cout <<"seq_length = "<< seq_l << std::endl;

    faInit(data);

    float con = pow(2 * EIGEN_PI, -units / 2.0);
    int tailIdx = seq_l - 1;

    Eigen::VectorXf temp1(units);
    Eigen::VectorXf temp2(units);
    float temp3 = 0;
    Eigen::VectorXf temp4(units);
    Eigen::MatrixXf invP(units, units);
    Eigen::VectorXf CP(units);
    Eigen::VectorXf Kcur(units);
    float KC(units);
    Eigen::VectorXf yDiff(units);
    Eigen::VectorXf xCur(seq_l);
    Eigen::VectorXf pCur(seq_l);

    Eigen::VectorXf pPre(seq_l);
    float detiP;
    float xPre = x0;

    float A1 = 0;
    float A2 = 0;
    float A3 = 0;
    float Ptsum = 0;
    float pt = 0;
    Eigen::VectorXf YX = Eigen::VectorXf::Zero(units);
    Eigen::VectorXf Xfin(seq_l);
    Eigen::VectorXf Pfin(seq_l);
    Eigen::VectorXf J = Eigen::VectorXf::Zero(seq_l);
    float Pcov;
    float T1;
    Eigen::VectorXf YY = tSeq.cwiseProduct(tSeq).colwise().sum() / seq_l;

    float lik = 0;
    float oldlik;
    float likbase;
    for (int i = 0; i < emMaxIter; i++)
    {
        //-----E step-----
        oldlik = lik;
        //---kalmansmooth
        lik = 0;
        pPre(0) = q0;
        xPre = x0;
        //forward pass
        for (int t = 0; t < seq_l; t++)
        {
            temp1 = c.cwiseQuotient(r);
            temp2 = temp1*pPre(t);
            temp3 = c.dot(temp2);
            temp4 = temp1*(1.0 / (temp3 + 1));
            invP = r.cwiseInverse().asDiagonal();
            invP = invP - temp2*temp4.transpose();
            CP = temp1 - temp3*temp4;
            Kcur = pPre(t)*CP;
            KC = Kcur.transpose()*c;
            yDiff = tSeq.row(t) - xPre*c.transpose();
            xCur(t) = xPre + yDiff.transpose()*Kcur;
            pCur(t) = pPre(t) - KC*pPre(t);
            if (t < seq_l - 1)
            {
                xPre = xCur(t)*a;
                pPre(t + 1) = a*pCur(t)*a + q;
            }

            //calculate likelihood
            detiP = sqrt(invP.determinant());

            lik = lik + log(detiP) - 0.5*(yDiff.transpose().cwiseProduct(yDiff.transpose()*invP)).sum();

            /* matlab exception handle code
                if (isreal(detiP) & detiP>0)
                lik=lik+N*log(detiP)-0.5*sum(sum(Ydiff.*(Ydiff*invP)));
                else
                problem=1;
                end;
                */
        }
        lik = lik + seq_l*log(con);

        //backward pass
        A1 = 0;
        A2 = 0;
        A3 = 0;
        Ptsum = 0;
        YX.setZero();
        Xfin(tailIdx) = xCur(tailIdx);
        Pfin(tailIdx) = pCur(tailIdx);
        pt = Pfin(tailIdx) + Xfin(tailIdx)*Xfin(tailIdx);
        A2 = -pt;
        Ptsum = pt;
        YX = tSeq.row(tailIdx)*Xfin(tailIdx);
        for (int t = tailIdx - 1; t >= 0; t--)
        {
            J(t) = pCur(t)*a / pPre(t + 1);
            Xfin(t) = xCur(t) + (Xfin(t + 1) - xCur(t)*a)*J(t);
            Pfin(t) = pCur(t) + J(t)*(Pfin(t + 1) - pPre(t + 1))*J(t);
            pt = Pfin(t) + Xfin(t)*Xfin(t);
            Ptsum = Ptsum + pt;
            YX = YX + tSeq.row(t).transpose()*Xfin(t);
        }
        A3 = Ptsum - pt;
        A2 = Ptsum + A2;
        Pcov = (1 - KC)*a*pCur(tailIdx - 1);
        A1 = A1 + Pcov + Xfin(tailIdx)*Xfin(tailIdx - 1);
        for (int t = tailIdx - 1; t > 0; t--)
        {
            Pcov = (pCur(t) + J(t)*(Pcov - a*pCur(t)))*J(t - 1);
            A1 = A1 + Pcov + Xfin(t)*Xfin(t - 1);
        }
        if (i <= 2)
        {
            likbase = lik;
        }
        else if (lik < oldlik)
        {
            std::cout << "violation" << " lik=" << lik << " oldlik="<<oldlik << std::endl;
        }
        else if (lik - likbase < (1 + emTol)*(oldlik - likbase) || std::isinf(lik))
        {
            break;
        }

        //----M step-----
        //Re-estimate A,C,Q,R,x0,P0;
        //x0 = Xfin(0);
        x0 = Xfin(seq_l-1);
        //T1 = Xfin(0) - x0;
        //q0 = Pfin(0);
        q0 = Pfin(seq_l-1);
        c = YX / Ptsum;
        r = YY - (c*YX.transpose()).diagonal() / seq_l;
        a = A1 / A2;
        q = (1.0 / (seq_l - 1))*(A3 - a*A1);
        /*
            if (det(Q)<0)
            fprintf('Q problem\n');
            end;
            */
        xsm = Xfin;
        Vsm = Pfin;
        std::cout << "TLDS EM iter="<<i<<std::endl;
    }
    init =true;
}

float TLDSdecoder::decode(const Eigen::VectorXf& data)
{
    if (!init) return 0;
    int v_units = data.rows();
    if (v_units != units)
    {
        std::cout << "TLDSdecoder:vector length different from model"
                  << "v_units=" << v_units << "model units=" << units << std::endl;
        return 0;
    }

    //units = 14;

    float x_pred, q_pred, q_filt, x_filt;
    Eigen::VectorXf K(units);
    Eigen::VectorXf y_pred(units);
    //prediction
    x_pred = a*x0;
    q_pred = a*q0*a + q;
    y_pred = c*x_pred + d;
    Eigen::MatrixXf rr = r.asDiagonal();

    K = q_pred*c.transpose()*((c*q_pred*c.transpose()) + rr).inverse();

    //filtering
    x_filt = (K.dot(data - y_pred) + x_pred);
    q_filt = (1 - K.dot(c))*q_pred;
    x0 = x_filt;
    q0 = q_filt;

    return x_filt;
}