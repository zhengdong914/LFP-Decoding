#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <Eigen/Dense>
#include "mtpsd.h"
#include "algorithm.h"
#include <sys/time.h>
#include <pthread.h>

using namespace std;
using Eigen::MatrixXd;
using Eigen::MatrixXf;
using Eigen::VectorXf;
using Eigen::Map;

//#define TRAINING
//#define MULTI_THREAD
#define CHANNELS 2  //LFP Channels
#define FEATURES 2*CHANNELS  //Each channel has 2 features(theta and high-gammma)
#define STEPS 25   //if sampling rate is 1kHz, 25 refres moving window step is 25ms


struct Signal{
    vector<double> signal[CHANNELS];
};

struct Model{
    float f[FEATURES];  //mean of the features based on training data
    float a;
    float q;
    float x0;
    float q0;
    float c[FEATURES];
    float d[FEATURES];
    float r[FEATURES];
    float mb;   // mean of baseline
    float sb;   // std of baseline
};

struct Inargs{
    int id;   //thread ID
    double* data;
    mtpsd_workspace params;
    vector<double> feats[CHANNELS];
};

vector<double> feature_collect(double* signal, mtpsd_workspace params);
Signal load_data();
void save_model(TLDSdecoder Decoder, string filename, Model& model);
void load_model(TLDSdecoder& Decoder, string filename, VectorXf& mm);
void training_model();
void* run_thread(void* args);

VectorXf seq(FEATURES);
pthread_mutex_t mutex;

int main()
{
#ifdef TRAINING
    training_model();
#else
    TLDSdecoder Decoder;
    VectorXf mm(FEATURES);

    string filename="/home/xiaozd/CLionProjects/LFP_Decoding/model.dat";   //testing data path
    load_model(Decoder, filename, mm);
    Decoder.setInit();
    Decoder.setUnits(FEATURES);

    cout << " ==========================Model Information=========================== " << endl;
    cout << " a = " << Decoder.getA() << " q = " << Decoder.getQ() << " x0 = " << Decoder.getX0() << " q0 = " << Decoder.getQ0() << endl;
    cout << " C = " << Decoder.getC().transpose() << endl;
    cout << " d = " << Decoder.getD().transpose() << endl;
    cout << " r = " << Decoder.getR().transpose() << endl;
    cout << " ================================End=================================== " << endl;


    Signal signal;
    vector<double> LFP_signal[CHANNELS];
    vector<double> feat[CHANNELS];

    signal = load_data();
    for(int i=0;i<CHANNELS;i++){
        LFP_signal[i] = signal.signal[i];
    }

    mtpsd_workspace params;
    params.n=250;   //window length
    params.nW=3;
    params.weight_method=EIGEN;
    params.N=512;
    params.K=2*params.nW-1;
    params.Fs=1000;  //sampling rate
    params.remove_mean= false;

    int num = (LFP_signal[0].size()-params.n-1)/STEPS;
    VectorXf z_score(num);
    VectorXf confidence(num);

#ifdef MULTI_THREAD
    pthread_t tid[CHANNELS];
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
    pthread_mutex_init(&mutex, NULL);
#endif

    struct timeval t1,t2;
    gettimeofday(&t1,NULL);
    Inargs *input[CHANNELS];
    for(int i=0; i<num; i++) {
        for(int j=0;j<CHANNELS;j++){
#ifdef MULTI_THREAD
            input[j] = new Inargs();
            input[j]->id = j;
            input[j]->data = &LFP_signal[j][i*STEPS];
            input[j]->params = params;
            int ret;
            ret = pthread_create(&tid[j],&attr,run_thread,input[j]);
            if(ret!=0) cout<<"Create thread faild!  ID =  " << j <<endl;
            //delete input;
            //pthread_join(tid[j],NULL);
#else
            double* sig = &LFP_signal[j][i*STEPS];
            feat[j] = feature_collect(sig,params);
#endif
        }

        /*
        for(int j=0;j<CHANNELS;j++) {
           pthread_join(tid[j],NULL);
        }
        pthread_attr_destroy(&attr);
        pthread_mutex_destroy(&mutex);*/
#ifdef MULTI_THREAD
        seq = seq - mm;   //zero mean
#else
        seq[0] = feat[0][0]-mm[0];
        seq[1] = feat[0][1]-mm[1];
        seq[2] = feat[1][0]-mm[2];
        seq[3] = feat[1][1]-mm[3];
#endif
        //filtering
        float z;
        z = Decoder.decode(seq);
        z_score[i] = Decoder.getZscore(z);
        confidence[i] = Decoder.getConf(2);    //Confidence Interval: 2
    }
    gettimeofday(&t2,NULL);
    float cost = t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec)/1000000.0;
    cout << "Num: " << num << "  Time Cost: = " << cost << "(s)" << " average time  " << 1000*cost/num << "(ms)" << endl;
    cout << z_score << endl;
#endif
    return 0;
}

void* run_thread(void* args)
{
    Inargs *input = (Inargs *)args;
    cout << "Data = " << *(input->data) << endl;
    input->feats[input->id] = feature_collect(input->data, input->params);
    //pthread_mutex_lock(&mutex);
    seq[2*input->id] = input->feats[input->id][0];
    seq[2*input->id+1] = input->feats[input->id][1];
    //pthread_mutex_unlock(&mutex);
    cout << "Thread#: " << input->id << " Features: " << input->feats[input->id][0] << ", " << input->feats[input->id][1] << endl;
    pthread_exit(0);
}

Signal load_data()
{
    MatrixXf data(12000,2);
    Signal signal;
    ifstream infile("/home/xiaozd/CLionProjects/LFP_Decoding/sample.csv", ios::in);
    string lineStr;
    int m=0;
    int n;
    while(getline(infile, lineStr))
    {
        n=0;
        stringstream ss(lineStr);
        string str;
        while(getline(ss,str,','))
        {
            stringstream sss(str);
            sss >> data(m,n);
            n++;
        }
        m++;
    }
    for(int l=0;l<CHANNELS;l++)
        for(int k=0;k<data.rows();k++)
            signal.signal[l].push_back(data(k, l));

    //cout << "m =  " << m << "  n = " << n << endl;
    //cout << data(0,0) << "   " << data(0,1) << endl;
    return signal;
}

void load_model(TLDSdecoder& Decoder, string filename, VectorXf& mm)
{
    Model model;
    ifstream is(filename, ios_base::in | ios_base::binary);
    if (is){
        is.read(reinterpret_cast<char *>(&model), sizeof(model));
    }else{
        cout << "ERROR: Cannot open file ! " << endl;
    }
    is.close();
    Decoder.setA(model.a);
    Decoder.setQ(model.q);
    Decoder.setQ0(model.q0);
    Decoder.setX0(model.x0);
    Decoder.set_meanBase(model.mb);
    Decoder.set_stdBase(model.sb);
    VectorXf c(FEATURES),d(FEATURES),r(FEATURES);
    for(int i=0;i<FEATURES;i++){
        c[i] = model.c[i];
        d[i] = model.d[i];
        r[i] = model.r[i];
        mm[i] = model.f[i];
    }
    Decoder.setC(c);
    Decoder.setD(d);
    Decoder.setR(r);

}

void save_model(TLDSdecoder Decoder, string filename, Model& model)
{
    model.a = Decoder.getA();
    model.q = Decoder.getQ();
    model.x0 = Decoder.getX0();
    model.q0 = Decoder.getQ0();
    model.mb = Decoder.get_meanBase();
    model.sb = Decoder.get_stdBase();
    VectorXf c,d,r;
    c = Decoder.getC();
    d = Decoder.getD();
    r = Decoder.getR();
    for(int i=0;i<FEATURES;i++) {
        model.c[i] = c[i];
        model.d[i] = d[i];
        model.r[i] = r[i];
    }
    ofstream os(filename, ios_base::out | ios_base::binary);
    os.write(reinterpret_cast<char *>( &model), sizeof(model));
    os.close();
}

vector<double> feature_collect(double* signal, mtpsd_workspace params)
{
    mtpsd<double> spectrum(signal, params);
    try{
        spectrum.compute();
    }
    catch(ERR e){
        printf("ERROR: %s\n", e.getmsg());
    }

    vector<double> Power, Freq, Feat;
    for(int i=0;i<spectrum.length();i++)
    {
        Power.push_back(spectrum(i));
        Freq.push_back(spectrum.freq(i));
        //cout << "Ind: " << i << "  S= " << Power[i] << "  , f=" << Freq[i] << "Hz"<< endl;
    }
    // freq:  0, 1.95, 3.91, 5.85, 7.81, 9.77, 11.72, ...

    double feat1,feat2;
    feat1 = (Power[2]+Power[3]+Power[4])/3;
    Feat.push_back(feat1);
    feat2 = 0.0;
    for(int i=31;i<52;i++)
    {
        feat2 = feat2+Power[i];
    }
    Feat.push_back(feat2/21);
    //cout << Feat[0] << "   " << Feat[1] << endl;

    return Feat;
}

void training_model()
{
    Model model;
    Signal signal;
    vector<double> LFP_signal[CHANNELS];
    vector<double> feat[CHANNELS];

    signal = load_data();
    for(int i=0;i<CHANNELS;i++){
        LFP_signal[i] = signal.signal[i];
    }

    mtpsd_workspace params;
    params.n=250;   //window length
    params.nW=3;
    params.weight_method=EIGEN;
    params.N=512;
    params.K=2*params.nW-1;
    params.Fs=1000;  //sampling rate
    params.remove_mean= false;

    MatrixXf train_data((LFP_signal[0].size()-params.n-1)/STEPS, FEATURES);
    for(int i=0; i<train_data.rows(); i++) {
        for(int j=0;j<CHANNELS;j++){
            double* sig = &LFP_signal[j][i*STEPS];
            feat[j] = feature_collect(sig,params);
        }
        train_data(i,0) = feat[0][0];
        train_data(i,1) = feat[0][1];
        train_data(i,2) = feat[1][0];
        train_data(i,3) = feat[1][1];
        //cout << train_data;
    }
    // demean
    VectorXf mm;
    mm = train_data.colwise().mean();
    train_data = train_data.rowwise() - mm.transpose();
    for(int i=0;i<mm.size();i++)
        model.f[i] = mm[i];
    TLDSdecoder Decoder;

    Decoder.train(train_data);

    VectorXf z_filter(train_data.rows());
    for(int i=0;i<train_data.rows();i++)
    {
        VectorXf seq(FEATURES);
        float z;
        for(int j=0;j<FEATURES;j++)
            seq(j) = train_data(i, j);

        z = Decoder.decode(seq);
        z_filter[i] = z;
    }
    Map<VectorXf> base(z_filter.data(), z_filter.size()/2);
    Decoder.updateBaseline(base);
    cout << "mean : " << Decoder.get_meanBase() << endl;

    string filename="/home/xiaozd/CLionProjects/LFP_Decoding/model.dat";
    save_model(Decoder,filename, model);
}