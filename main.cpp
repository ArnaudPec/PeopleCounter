#include <iostream>
#include <iomanip>
#include <vector>

#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <ctype.h>

#include "opencv2/opencv.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/tracking.hpp"
#include "opencv2/tracking/tldDataset.hpp"

#define SIZEX 640
#define SIZEY 480

using namespace std;
using namespace cv;

CascadeClassifier face_cascade;
Ptr<BackgroundSubtractor> pMOG2;

MultiTracker trackers;
void usage(char *s);

char *filename;

struct coord {
    int x;
    int y;
};

/*
 * Class which represent a person as an object to be tracked with coordinates
 */
class Person
{
public:
    //virtual ~Person ();
    Person (int id, int xi, int yi, int max_agei);
    void updateCoord (int xn, int yn);
    void incAge();

private:
    int i;
    int x;
    int y;
    int max_age;
    int age;
    string dir;
    bool done;
    vector<coord> tracks;
};

/* Constructor */
Person::Person (int id, int xi, int yi, int max_agei) {
    i = id;
    x = xi;
    y = yi;
    max_age = max_agei;
}

/*
 * Update the position and the "tracks":
 *     - Last position is stored in tracks
 *     - Position is updated with current values
 *     - age is set to 0
 */
void Person::updateCoord (int xn, int yn) {
    struct coord ncord;
    ncord.x = x;
    ncord.y = y;
    tracks.push_back(ncord);
    x = xn;
    y = yn;
    age = 0;
}

/*
 * Increment age of the last detection
 */
void Person::incAge()
{
    age++;
    if (age > max_age) {
       done = true;
    }
}

vector<Person> persons;

void detect(Mat frame) {

    Mat frame_gray, bgless, bin, kernelo, kernelc;
    Mat binc, bino;
    Size sz = frame.size();
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;

    char str[20];
    double area, areaTh = 0;
    Rect r;

    kernelo = Mat::ones(3, 3, CV_8UC1);
    kernelc = Mat::ones(11, 11, CV_8UC1);

    pMOG2->apply(frame, bgless);
    threshold(bgless, bin, 200, 255, THRESH_BINARY);

    //opening erode dilate
    morphologyEx(bin, bino, 0, kernelo);

    // closing dilate erode
    morphologyEx(bino, binc, 1, kernelc);

    // Put some text
    sprintf(str,"%d pers", 0);

    putText(frame, str, Point2f(50,50), FONT_HERSHEY_SIMPLEX, 1,
            Scalar(0,0,255,255), 3);

    // Draw a line
    line(frame, Point2f(0, sz.height/2), Point2f(sz.width, sz.height/2),
         Scalar(0, 0, 0), 8, 2);

    // Draw green contours
    findContours(binc, contours, hierarchy, CV_RETR_EXTERNAL,
                 CV_CHAIN_APPROX_NONE, Point(0, 0));

    for(int i = 0; i < contours.size(); i++)
    {
        drawContours(frame, contours, (int)i, Scalar(0,0,255,255), 2,
                     8, hierarchy, 0, Point());

        area=contourArea(contours[i],false);

        if(area>areaTh) {
            areaTh = area;
            r = boundingRect(contours[i]);
            rectangle(frame, r, Scalar(0,0,255,255), 2, 8);
        }
    }

    imshow("Frame 1", frame);
    imshow("Post", binc);
    //imshow("Background less", bgless);
    //imshow("Binarized", bin);
}

int main(int argc, char *argv[])
{

    Ptr<TrackerKCF> tracker = TrackerKCF::create();
    VideoCapture camera("peopleCounter.avi");
    //VideoCapture camera(0);

    camera.set(CV_CAP_PROP_FRAME_WIDTH, SIZEX);
    camera.set(CV_CAP_PROP_FRAME_HEIGHT, SIZEY);

    if (!camera.isOpened())
    {
        cout << "Cannot open camera" << endl;
        return 1;
    }

    pMOG2 = createBackgroundSubtractorMOG2();

    for(;;)
    {
        Mat frame;
        camera.grab();
        //
        camera.retrieve(frame);
        tracker->init(frame, bbox);
        detect(frame);
        //

        if (waitKey(30) == 27) break;
    }
    camera.release();

    return 0;

    //int opt;

    //if(argc == 1)
    //{
    //usage(argv[0]);
    //exit(1);
    //}

    //// x: signifie que x attend un paramètre
    //while ((opt = getopt(argc,argv,"hc:m:t:")) != EOF){

    //switch(opt)
    //{
    //case 'h':
    //usage(argv[0]);
    //exit(1);

    //case 'c':
    ////filename = optarg;
    ////cout << "Nouveau net : "<< filename <<endl ;
    ////createNet(filename);
    ////return 0;

    //case '?':
    //if (optopt == 'c' || optopt == 'm' || optopt == 't')
    //fprintf (stderr,
    //"Option -%c requiert un argument.\n",
    //optopt);
    //else if (isprint (optopt))
    //fprintf (stderr,
    //"Option inconnue `-%c'.\n",
    //optopt);
    //else
    //fprintf (stderr,
    //"Caractere option inconnue`\\x%x'.\n",
    //optopt);
    //return 1;

    //default:
    //usage(argv[0]);

    //}
    //}

    return 0;
}

void usage(char *s)
{
  cout<<"Usage:   "<<s<<" [-option] [argument]"<<endl;
  cout<<"option:  "<<"-h affiche l'aide"<<endl;
}
