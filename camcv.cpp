/////////////////////////////////////////////////////////////
// Many source code lines are copied from RaspiVid.c
// Copyright (c) 2012, Broadcom Europe Ltd
/////////////////////////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <memory.h>

//new
#include <cv.h>
#include <highgui.h>
#include "time.h"

extern "C" {
	#include "bcm_host.h"
	#include "interface/vcos/vcos.h"
	
	#include "interface/mmal/mmal.h"
	#include "interface/mmal/mmal_logging.h"
	#include "interface/mmal/mmal_buffer.h"
	#include "interface/mmal/util/mmal_util.h"
	#include "interface/mmal/util/mmal_util_params.h"
	#include "interface/mmal/util/mmal_default_components.h"
	#include "interface/mmal/util/mmal_connection.h"
	
	#include "RaspiCamControl.h"
	#include "RaspiPreview.h"
	#include "RaspiCLI.h"
	
	////////////////////////////////////////////////////////
	// PWM STUFF
	////////////////////////////////////////////////////////
    #include <wiringPi.h>
	////////////////////////////////////////////////////////
	// END PWM STUFF
	////////////////////////////////////////////////////////
}

#include <semaphore.h>
#include <cmath>


const int ABORT_INTERVAL = 100; // ms
// OPENCV
#include <iostream>
#include <fstream>
#include <sstream>
#include "time.h"
#include <vector>
#include <algorithm>
bool lastBlackLine = true;
double oranr = 0.5;
double oranfr = 0.8;
double minSpeed = 230;
bool isSqON = true;
bool printCMD = false;
bool printMaxVel = false;
bool printRobotIsOff = false;
bool printChangeMotor = true;
int rightMotorPinPWM = 1;
int leftMotorPinPWM = 23;
int rightMotorPin2 = 5;
int leftMotorPin2 = 2;
int buttonPin = 0;

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

// some constants to manage nb of people to learn+ id of people 
#define MAX_PEOPLE 		4
#define P_PIERRE		0
#define P_NATACHA		1
#define P_LISA			3
#define P_MONA			2

// for debug and trace
#define TRACE 1
#define DEBUG_MODE 0
#define DEBUG if (DEBUG_MODE==1)

CascadeClassifier face_cascade; 
CvPoint Myeye_left;
CvPoint Myeye_right;
Eigenfaces model;
string fn_haar;
string fn_csv;
int im_width;		// image width
int im_height;		// image height
int PREDICTION_SEUIL ;
char key;	

Mat _gray,__gray,gray,frame,face,face_resized;
vector<Mat> images;
vector<int> labels;

// name of people
string  people[MAX_PEOPLE];

// nb of times RPI talks to one guy
int nbSpeak[MAX_PEOPLE];

int bHisto;
vector< Rect_<int> > faces;

// nb of picture learnt by people
int nPictureById[MAX_PEOPLE];
///////////////////////

/// Camera number to use - we only have one camera, indexed from 0.
#define CAMERA_NUMBER 0

// Standard port setting for the camera component
#define MMAL_CAMERA_PREVIEW_PORT 0
#define MMAL_CAMERA_VIDEO_PORT 1
#define MMAL_CAMERA_CAPTURE_PORT 2

// Video format information
#define VIDEO_FRAME_RATE_NUM 30
#define VIDEO_FRAME_RATE_DEN 1

/// Video render needs at least 2 buffers.
#define VIDEO_OUTPUT_BUFFERS_NUM 3

// Max bitrate we allow for recording
const int MAX_BITRATE = 30000000; // 30Mbits/s


// variable to convert I420 frame to IplImage
int nCount=0;
IplImage *py, *pu, *pv, *pu_big, *pv_big, *image,* dstImage;
		
		
int mmal_status_to_int(MMAL_STATUS_T status);

/** Structure containing all state information for the current run
 */
typedef struct
{
   int timeout;                        /// Time taken before frame is grabbed and app then shuts down. Units are milliseconds
   int width;                          /// Requested width of image
   int height;                         /// requested height of image
   int bitrate;                        /// Requested bitrate
   int framerate;                      /// Requested frame rate (fps)
   int graymode;			/// capture in gray only (2x faster)
   int immutableInput;      /// Flag to specify whether encoder works in place or creates a new buffer. Result is preview can display either
                                       /// the camera output or the encoder output (with compression artifacts)
   RASPIPREVIEW_PARAMETERS preview_parameters;   /// Preview setup parameters
   RASPICAM_CAMERA_PARAMETERS camera_parameters; /// Camera setup parameters

   MMAL_COMPONENT_T *camera_component;    /// Pointer to the camera component
   MMAL_COMPONENT_T *encoder_component;   /// Pointer to the encoder component
   MMAL_CONNECTION_T *preview_connection; /// Pointer to the connection from camera to preview
   MMAL_CONNECTION_T *encoder_connection; /// Pointer to the connection from camera to encoder

   MMAL_POOL_T *video_pool; /// Pointer to the pool of buffers used by encoder output port
   
} RASPIVID_STATE;

/** Struct used to pass information in encoder port userdata to callback
 */
typedef struct
{
   FILE *file_handle;                   /// File handle to write buffer data to.
   VCOS_SEMAPHORE_T complete_semaphore; /// semaphore which is posted when we reach end of frame (indicates end of capture or fault)
   RASPIVID_STATE *pstate;            /// pointer to our state in case required in callback
} PORT_USERDATA;

/////////////////////////////////////////////////
// trace if TRACE==1
/////////////////////////////////////////////////
void trace(string s)
{
	if (TRACE==1)
	{
		cout<<s<<"\n";
	}
}

// default status
static void default_status(RASPIVID_STATE *state)
{
   if (!state)
   {
      vcos_assert(0);
      return;
   }

   // Default everything to zero
   memset(state, 0, sizeof(RASPIVID_STATE));

   // Now set anything non-zero
   state->timeout 			= -1; //65000;     // capture time : here 65 s
   state->width 			= 320;      // use a multiple of 320 (640, 1280)
   state->height 			= 240;		// use a multiple of 240 (480, 960)
   state->bitrate 			= 17000000; // This is a decent default bitrate for 1080p
   state->framerate 		= VIDEO_FRAME_RATE_NUM;
   state->immutableInput 	= 1;
   state->graymode 			= 1;		//gray by default, much faster than color (0) // mandatory for line following
   
   // Setup preview window defaults
   raspipreview_set_defaults(&state->preview_parameters);

   // Set up the camera_parameters to default
   raspicamcontrol_set_defaults(&state->camera_parameters);
}

int change_limit = 0;
int old_trans = 0, old_rot = 0;
bool isLastRight = false;
void changeMotor(double tran, double rot){
	if(abs(old_trans - tran) <= change_limit && abs(old_rot - rot) <= change_limit * oranr && tran != 0)
		return;
	old_trans = tran;
	old_rot = rot;
	
	if(printChangeMotor){
		int inputstate = digitalRead(buttonPin);
		if(inputstate == HIGH)
			cout << tran << " " << rot << endl;
	}
		
	double v1 = tran + rot;
	double v2 = tran - rot;
	
	bool isLeftForw = v1 > 0;
	v1 = abs(v1);
	bool isRightForw = v2 > 0;
	v2 = abs(v2);
	if(v1 < 0) v1 = 0;
	if(v1 > 255) v1 = 255;
	if(v2 < 0) v2 = 0;
    if(v2 > 255) v2 = 255;
    v1 *= 4.0;
    v2 *= 4.0;
	if(!isLeftForw){
		pwmWrite (leftMotorPinPWM, v2);
		digitalWrite (leftMotorPin2, LOW);
		cout << "LPWM: " << v2 << " LOW" << endl;
	}
	else{
		pwmWrite (leftMotorPinPWM, 1023 - v2);
		digitalWrite (leftMotorPin2, HIGH);
		cout << "LPWM: " << 1023 - v2 << " HIGH" << endl;
	}
	
	if(!isRightForw){
		pwmWrite (rightMotorPinPWM, v1);
		digitalWrite (rightMotorPin2, LOW);
		cout << "RPWM: " << v1 << " LOW" << endl;
	}
	else{
		pwmWrite (rightMotorPinPWM, 1023 - v1);
		digitalWrite (rightMotorPin2, HIGH);
		cout << "RPWM: " << 1023 - v1 << " HIGH" << endl;
    }
}

int threhold = 60;

/**
 *  buffer header callback function for video
 *
 * @param port Pointer to port from which callback originated
 * @param buffer mmal buffer header pointer
 */
static void video_buffer_callback(MMAL_PORT_T *port, MMAL_BUFFER_HEADER_T *buffer)
{
   MMAL_BUFFER_HEADER_T *new_buffer;
   PORT_USERDATA *pData = (PORT_USERDATA *)port->userdata;

   if (pData)
   {
     
      if (buffer->length)
      {

	      mmal_buffer_header_mem_lock(buffer);
 
 		//
		// *** PR : OPEN CV Stuff here !
		//
		int w=pData->pstate->width;	// get image size
		int h=pData->pstate->height;
		int h4=h/4;
		
		memcpy(py->imageData,buffer->data,w*h);	// read Y
		
		if (pData->pstate->graymode==0)
		{
			memcpy(pu->imageData,buffer->data+w*h,w*h4); // read U
			memcpy(pv->imageData,buffer->data+w*h+w*h4,w*h4); // read v
	
			cvResize(pu, pu_big, CV_INTER_NN);
			cvResize(pv, pv_big, CV_INTER_NN);  //CV_INTER_LINEAR looks better but it's slower
			cvMerge(py, pu_big, pv_big, NULL, image);
	
			cvCvtColor(image,dstImage,CV_YCrCb2RGB);	// convert in RGB color space (slow)
			gray=cvarrToMat(dstImage);   
			//cvShowImage("camcvWin", dstImage );
			
		}
		else
		{	
			// for face reco, we just keep gray channel, py
			gray=cvarrToMat(py);  
			//cvShowImage("camcvWin", py); // display only gray channel
		}

////////////////////////////////
// LINE DETECTION START HERE
////////////////////////////////
	Size sz(160,120);
	
	resize(gray, _gray, sz);

    //_gray = gray.clone();
    
    if(lastBlackLine)
        _gray = _gray < threhold;
    else
        _gray = _gray >= threhold;
    
    __gray = _gray.clone();
    
    _gray /= 4;

    vector<vector<Point> > contours;
    findContours(__gray, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
    
    
    vector<vector<int> > crossingPoints;
    vector<double> centerOfMassesX;
    int y = sz.height;
    do{
        y -= 5;
        crossingPoints.clear();
        centerOfMassesX.clear();
        for(int i = 0; i < contours.size(); i++){
            bool drawn = false;
            double sum = 0;
            for(int j = 0; j < contours[i].size(); j++){
                sum += contours[i][j].x;
                if(contours[i][j].y == y /*&& contours[i][j].x < 280 &&  contours[i][j].x > 50*/){
                    //_gray.at<char>(Point(contours[i][j].x, contours[i][j].y)) = 255;

                    if(!drawn){
                        crossingPoints.push_back(vector<int>());
                        drawContours(_gray, contours, i, Scalar(255 / 2), -1);
                        drawn = true;
                    }
                    crossingPoints[crossingPoints.size() - 1].push_back(contours[i][j].x);
                }
            }
            if(drawn){
                sum /= contours[i].size();
                centerOfMassesX.push_back(sum);
            }
        }
    } while(crossingPoints.size() == 0 && y >= sz.height *2/3);
	
	
    for(int i = 1; i < sz.width; i++)
        _gray.at<char>(Point(i, y)) = 0;
	
	int inputstate = digitalRead(buttonPin);
	if(inputstate == LOW){
		if(printRobotIsOff)
			cout << "0ROBOT Is OFF" << endl;
		changeMotor(0, 0);
	}
	else{
		if(crossingPoints.size() != 0){
			vector<vector<int> > crossParts;
			vector<int> crossPartsToCrossingPointsMap;
			for(int i = 0; i < crossingPoints.size(); i++){
				sort(crossingPoints[i].begin(),crossingPoints[i].end());

				for(int j = 0; j < crossingPoints[i].size() - 1; j += 2){
					crossParts.push_back(vector<int>());
					crossParts[crossParts.size()-1].push_back(crossingPoints[i][j]);
					crossParts[crossParts.size()-1].push_back(crossingPoints[i][j + 1]);

					crossPartsToCrossingPointsMap.push_back(i);

					for(int k = crossingPoints[i][j]; k < crossingPoints[i][j + 1]; k++)
						_gray.at<char>(Point(k, y)) = 120;
				}
			}

			
			if(crossParts.size() != 0){
				bool flag = true;
				bool isLineColourChanged = false;
				int lineSize = 0;
				for(int i = 0; i < crossParts.size(); i++)
					lineSize += (crossParts[i][1] - crossParts[i][0]);
				if(lineSize > 250)
					isLineColourChanged = true;

				if(isLineColourChanged){
					_gray = gray.clone();
					if(lastBlackLine)
						_gray = _gray >= threhold;
					else
						_gray = _gray < threhold;

					lastBlackLine = !lastBlackLine;

					__gray = _gray.clone();

					_gray /= 4;

					contours.clear();
					findContours(__gray, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);


					crossingPoints.clear();
					centerOfMassesX.clear();
					int y = sz.height;

					do{
						y -= 5;
						crossingPoints.clear();
						for(int i = 0; i < contours.size(); i++){
							bool drawn = false;
							double sum = 0;
							for(int j = 0; j < contours[i].size(); j++){
								sum += contours[i][j].x;
								if(contours[i][j].y == y){
									//_gray.at<char>(Point(contours[i][j].x, contours[i][j].y)) = 255;

									if(!drawn){
										crossingPoints.push_back(vector<int>());
										drawContours(_gray, contours, i, Scalar(255 / 2), -1);
										drawn = true;
									}
									crossingPoints[crossingPoints.size() - 1].push_back(contours[i][j].x);
								}
							}
							if(drawn){
								sum /= contours[i].size();
								centerOfMassesX.push_back(sum);
							}
						}
					} while(crossingPoints.size() == 0 && y >= sz.height *2/3);


					for(int i = 1; i < sz.width; i++)
						_gray.at<char>(Point(i, y)) = 0;


					if(crossingPoints.size() != 0){
						crossParts.clear();
						crossPartsToCrossingPointsMap.clear();
						for(int i = 0; i < crossingPoints.size(); i++){
							sort(crossingPoints[i].begin(),crossingPoints[i].end());

							for(int j = 0; j < crossingPoints[i].size() - 1; j += 2){
								crossParts.push_back(vector<int>());
								crossParts[crossParts.size()-1].push_back(crossingPoints[i][j]);
								crossParts[crossParts.size()-1].push_back(crossingPoints[i][j + 1]);

								crossPartsToCrossingPointsMap.push_back(i);

								for(int k = crossingPoints[i][j]; k < crossingPoints[i][j + 1]; k++)
									_gray.at<char>(Point(k, y)) = 120;
							}
						}
					}
					else{
						flag = false;
						if(isLastRight){
							changeMotor(0.0, -255.0*oranfr);
							if(printCMD)
								cout << "2cmd trans=" << 0 << " rot=" << -255*oranr << " end" << endl;
						}
						else{
							changeMotor(0.0, 255.0*oranfr);
							if(printCMD)
								cout << "2cmd trans=" << 0 << " rot=" << 255*oranr << " end" << endl;
						}
					}
				}

				if(flag){
					int center = 0;
					int mindiff = 9999;
					int idx = 0;
					for(int i = 0; i < crossParts.size(); i++){
						int c = (crossParts[i][0] + crossParts[i][1]) / 2;
						if (abs(c - sz.width / 2) < mindiff){
							center = c;
							idx = i;
							mindiff = abs(c - sz.width / 2);
						}
					}
					for(int i = crossParts[idx][0]; i < crossParts[idx][1]; i++)
						_gray.at<char>(Point(i, y)) = 255;


					double trans, rot = 0;
					if(isSqON){
						int sgn = 1;
						if(center < sz.width / 2) sgn = -1;
						rot = sgn * (center - sz.width / 2)*(center - sz.width / 2)*(255.0/((sz.width / 2)*(sz.width / 2)));
					}
					else
						rot = (center - (sz.width / 2))*(255.0/(sz.width / 2));
                    isLastRight = centerOfMassesX[crossPartsToCrossingPointsMap[idx]] - (sz.width / 2) > 0;

					double cmDiff = abs(centerOfMassesX[crossPartsToCrossingPointsMap[idx]] - sz.width / 2);
					double maxSpeed;
					if(cmDiff < 3)
						maxSpeed = 255;
					else if(cmDiff > 40)
						maxSpeed = minSpeed;
					else
						maxSpeed = minSpeed + (40 - cmDiff) / (40 - 3) * (255 - minSpeed);
					if(printMaxVel)
						cout << maxSpeed << endl;

					trans = maxSpeed - abs(rot * oranr * maxSpeed / 255);
					
					changeMotor(trans, rot*oranr);
					if(printCMD)
						cout << "1cmd trans=" << trans << " rot=" << rot*oranr << " end" << endl;
				}
			}
		}
		else{
			if(isLastRight){
				changeMotor(0.0, -255.0*oranfr);
				if(printCMD)
					cout << "2cmd trans=" << 0 << " rot=" << -255*oranfr << " end" << endl;
			}
			else{
				changeMotor(0.0, 255.0*oranfr);
				if(printCMD)
					cout << "2cmd trans=" << 0 << " rot=" << 255*oranfr << " end" << endl;
			}
		}
	}
	

/////////////////////////////
// END OF LINE DETECTION
/////////////////////////////
		
	// Show the result:
	imshow("camcvWin", _gray);
	imshow("camcvWin2", gray);
	key = (char) waitKey(1);
	nCount++;		// count frames displayed
		
         mmal_buffer_header_mem_unlock(buffer);
      }
      else vcos_log_error("buffer null");
      
   }
   else
   {
      vcos_log_error("Received a encoder buffer callback with no state");
   }

   // release buffer back to the pool
   mmal_buffer_header_release(buffer);

   // and send one back to the port (if still open)
   if (port->is_enabled)
   {
      MMAL_STATUS_T status;

      new_buffer = mmal_queue_get(pData->pstate->video_pool->queue);

      if (new_buffer)
         status = mmal_port_send_buffer(port, new_buffer);

      if (!new_buffer || status != MMAL_SUCCESS)
         vcos_log_error("Unable to return a buffer to the encoder port");
   }
    
}


/**
 * Create the camera component, set up its ports
 *
 * @param state Pointer to state control struct
 *
 * @return 0 if failed, pointer to component if successful
 *
 */
static MMAL_COMPONENT_T *create_camera_component(RASPIVID_STATE *state)
{
	MMAL_COMPONENT_T *camera = 0;
	MMAL_ES_FORMAT_T *format;
	MMAL_PORT_T *preview_port = NULL, *video_port = NULL, *still_port = NULL;
	MMAL_STATUS_T status;
	
	/* Create the component */
	status = mmal_component_create(MMAL_COMPONENT_DEFAULT_CAMERA, &camera);
	
	if (status != MMAL_SUCCESS)
	{
	   vcos_log_error("Failed to create camera component");
	   goto error;
	}
	
	if (!camera->output_num)
	{
	   vcos_log_error("Camera doesn't have output ports");
	   goto error;
	}
	
	video_port = camera->output[MMAL_CAMERA_VIDEO_PORT];
	still_port = camera->output[MMAL_CAMERA_CAPTURE_PORT];
	
	//  set up the camera configuration
	{
	   MMAL_PARAMETER_CAMERA_CONFIG_T cam_config =
	   {
	      { MMAL_PARAMETER_CAMERA_CONFIG, sizeof(cam_config) },
	      cam_config.max_stills_w = state->width,
	      cam_config.max_stills_h = state->height,
	      cam_config.stills_yuv422 = 0,
	      cam_config.one_shot_stills = 0,
	      cam_config.max_preview_video_w = state->width,
	      cam_config.max_preview_video_h = state->height,
	      cam_config.num_preview_video_frames = 3,
	      cam_config.stills_capture_circular_buffer_height = 0,
	      cam_config.fast_preview_resume = 0,
	      cam_config.use_stc_timestamp = MMAL_PARAM_TIMESTAMP_MODE_RESET_STC
	   };
	   mmal_port_parameter_set(camera->control, &cam_config.hdr);
	}
	// Set the encode format on the video  port
	
	format = video_port->format;
	format->encoding_variant = MMAL_ENCODING_I420;
	format->encoding = MMAL_ENCODING_I420;
	format->es->video.width = state->width;
	format->es->video.height = state->height;
	format->es->video.crop.x = 0;
	format->es->video.crop.y = 0;
	format->es->video.crop.width = state->width;
	format->es->video.crop.height = state->height;
	format->es->video.frame_rate.num = state->framerate;
	format->es->video.frame_rate.den = VIDEO_FRAME_RATE_DEN;
	
	status = mmal_port_format_commit(video_port);
	if (status)
	{
	   vcos_log_error("camera video format couldn't be set");
	   goto error;
	}
	
	// PR : plug the callback to the video port 
	status = mmal_port_enable(video_port, video_buffer_callback);
	if (status)
	{
	   vcos_log_error("camera video callback2 error");
	   goto error;
	}

   // Ensure there are enough buffers to avoid dropping frames
   if (video_port->buffer_num < VIDEO_OUTPUT_BUFFERS_NUM)
      video_port->buffer_num = VIDEO_OUTPUT_BUFFERS_NUM;


   // Set the encode format on the still  port
   format = still_port->format;
   format->encoding = MMAL_ENCODING_OPAQUE;
   format->encoding_variant = MMAL_ENCODING_I420;
   format->es->video.width = state->width;
   format->es->video.height = state->height;
   format->es->video.crop.x = 0;
   format->es->video.crop.y = 0;
   format->es->video.crop.width = state->width;
   format->es->video.crop.height = state->height;
   format->es->video.frame_rate.num = 1;
   format->es->video.frame_rate.den = 1;

   status = mmal_port_format_commit(still_port);
   if (status)
   {
      vcos_log_error("camera still format couldn't be set");
      goto error;
   }

	
	//PR : create pool of message on video port
	MMAL_POOL_T *pool;
	video_port->buffer_size = video_port->buffer_size_recommended;
	video_port->buffer_num = video_port->buffer_num_recommended;
	pool = mmal_port_pool_create(video_port, video_port->buffer_num, video_port->buffer_size);
	if (!pool)
	{
	   vcos_log_error("Failed to create buffer header pool for video output port");
	}
	state->video_pool = pool;

	/* Ensure there are enough buffers to avoid dropping frames */
	if (still_port->buffer_num < VIDEO_OUTPUT_BUFFERS_NUM)
	   still_port->buffer_num = VIDEO_OUTPUT_BUFFERS_NUM;
	
	/* Enable component */
	status = mmal_component_enable(camera);
	
	if (status)
	{
	   vcos_log_error("camera component couldn't be enabled");
	   goto error;
	}
	
	raspicamcontrol_set_all_parameters(camera, &state->camera_parameters);
	
	state->camera_component = camera;
	
	return camera;

error:

   if (camera)
      mmal_component_destroy(camera);

   return 0;
}

/**
 * Destroy the camera component
 *
 * @param state Pointer to state control struct
 *
 */
static void destroy_camera_component(RASPIVID_STATE *state)
{
   if (state->camera_component)
   {
      mmal_component_destroy(state->camera_component);
      state->camera_component = NULL;
   }
}


/**
 * Destroy the encoder component
 *
 * @param state Pointer to state control struct
 *
 */
static void destroy_encoder_component(RASPIVID_STATE *state)
{
   // Get rid of any port buffers first
   if (state->video_pool)
   {
      mmal_port_pool_destroy(state->encoder_component->output[0], state->video_pool);
   }

   if (state->encoder_component)
   {
      mmal_component_destroy(state->encoder_component);
      state->encoder_component = NULL;
   }
}

/**
 * Connect two specific ports together
 *
 * @param output_port Pointer the output port
 * @param input_port Pointer the input port
 * @param Pointer to a mmal connection pointer, reassigned if function successful
 * @return Returns a MMAL_STATUS_T giving result of operation
 *
 */
static MMAL_STATUS_T connect_ports(MMAL_PORT_T *output_port, MMAL_PORT_T *input_port, MMAL_CONNECTION_T **connection)
{
   MMAL_STATUS_T status;

   status =  mmal_connection_create(connection, output_port, input_port, MMAL_CONNECTION_FLAG_TUNNELLING | MMAL_CONNECTION_FLAG_ALLOCATION_ON_INPUT);

   if (status == MMAL_SUCCESS)
   {
      status =  mmal_connection_enable(*connection);
      if (status != MMAL_SUCCESS)
         mmal_connection_destroy(*connection);
   }

   return status;
}

/**
 * Checks if specified port is valid and enabled, then disables it
 *
 * @param port  Pointer the port
 *
 */
static void check_disable_port(MMAL_PORT_T *port)
{
   if (port && port->is_enabled)
      mmal_port_disable(port);
}

/**
 * Handler for sigint signals
 *
 * @param signal_number ID of incoming signal.
 *
 */
static void signal_handler(int signal_number)
{
	// Going to abort on all signals
	vcos_log_error("Aborting program\n");
	
	cout << "stopping Motor" << endl;
	changeMotor(0, 0);

	exit(255);
}

/**
 * main
 */
int main(int argc, const char **argv)
{
/////////////////////////////////
// PWM SETUP
/////////////////////////////////
	putenv("WIRINGPI_DEBUG=1");
	
	if (wiringPiSetup () == -1)
	{
		fprintf (stdout, "oops: WPSetup: %s\n", strerror (errno)) ;
		return 1 ;
    }
	pinMode (leftMotorPin2, OUTPUT);
	pinMode (rightMotorPin2, OUTPUT);
	pinMode (leftMotorPinPWM, PWM_OUTPUT);
	pinMode (rightMotorPinPWM, PWM_OUTPUT);
    pinMode (buttonPin, INPUT);
    
	cout << "pinMode - done" << endl;
/////////////////////////////////
// END PWM SETUP
/////////////////////////////////

    // Our main data storage vessel..
	RASPIVID_STATE state;
	
	MMAL_STATUS_T status;// = -1;
	MMAL_PORT_T *camera_video_port = NULL;
	MMAL_PORT_T *camera_still_port = NULL;
	MMAL_PORT_T *preview_input_port = NULL;
	MMAL_PORT_T *encoder_input_port = NULL;
	MMAL_PORT_T *encoder_output_port = NULL;
	
	time_t timer_begin,timer_end;
	double secondsElapsed;
	
	bcm_host_init();
	signal(SIGINT, signal_handler);

	// read default status
	default_status(&state);

	// init windows and OpenCV Stuff
	cvNamedWindow("camcvWin", CV_WINDOW_AUTOSIZE); 
	int w=state.width;
	int h=state.height;
	dstImage = cvCreateImage(cvSize(w,h), IPL_DEPTH_8U, 3);
	py = cvCreateImage(cvSize(w,h), IPL_DEPTH_8U, 1);		// Y component of YUV I420 frame
	pu = cvCreateImage(cvSize(w/2,h/2), IPL_DEPTH_8U, 1);	// U component of YUV I420 frame
	pv = cvCreateImage(cvSize(w/2,h/2), IPL_DEPTH_8U, 1);	// V component of YUV I420 frame
	pu_big = cvCreateImage(cvSize(w,h), IPL_DEPTH_8U, 1);
	pv_big = cvCreateImage(cvSize(w,h), IPL_DEPTH_8U, 1);
	image = cvCreateImage(cvSize(w,h), IPL_DEPTH_8U, 3);	// final picture to display

   
	// create camera
	if (!create_camera_component(&state))
	{
	   vcos_log_error("%s: Failed to create camera component", __func__);
	}
	else if ( (status = raspipreview_create(&state.preview_parameters)) != MMAL_SUCCESS )
	{
	   vcos_log_error("%s: Failed to create preview component", __func__);
	   destroy_camera_component(&state);
	}
	else
	{
		PORT_USERDATA callback_data;
		
		camera_video_port   = state.camera_component->output[MMAL_CAMERA_VIDEO_PORT];
		camera_still_port   = state.camera_component->output[MMAL_CAMERA_CAPTURE_PORT];
	   
		VCOS_STATUS_T vcos_status;
		
		callback_data.pstate = &state;
		
		vcos_status = vcos_semaphore_create(&callback_data.complete_semaphore, "RaspiStill-sem", 0);
		vcos_assert(vcos_status == VCOS_SUCCESS);
		
		// assign data to use for callback
		camera_video_port->userdata = (struct MMAL_PORT_USERDATA_T *)&callback_data;
        
        // init timer
  		time(&timer_begin); 

       
       // start capture
		if (mmal_port_parameter_set_boolean(camera_video_port, MMAL_PARAMETER_CAPTURE, 1) != MMAL_SUCCESS)
		{
		   	return 0;
		}
		
		// Send all the buffers to the video port
		
		int num = mmal_queue_length(state.video_pool->queue);
		int q;
		for (q=0;q<num;q++)
		{
		   MMAL_BUFFER_HEADER_T *buffer = mmal_queue_get(state.video_pool->queue);
		
		   if (!buffer)
		   		vcos_log_error("Unable to get a required buffer %d from pool queue", q);
		
			if (mmal_port_send_buffer(camera_video_port, buffer)!= MMAL_SUCCESS)
		    	vcos_log_error("Unable to send a buffer to encoder output port (%d)", q);
		}
		
		
		// Now wait until we need to stop
		if(state.timeout > 0)
			vcos_sleep(state.timeout);
		else
			while(true)
				vcos_sleep(ABORT_INTERVAL);
  
		//mmal_status_to_int(status);
		// Disable all our ports that are not handled by connections
		check_disable_port(camera_still_port);
		
		if (state.camera_component)
		   mmal_component_disable(state.camera_component);
		
		//destroy_encoder_component(&state);
		raspipreview_destroy(&state.preview_parameters);
		destroy_camera_component(&state);
	}
	
	if (status != 0)
		raspicamcontrol_check_configuration(128);
	
	time(&timer_end);  /* get current time; same as: timer = time(NULL)  */
	cvReleaseImage(&dstImage);
	cvReleaseImage(&pu);
	cvReleaseImage(&pv);
	cvReleaseImage(&py);
	cvReleaseImage(&pu_big);
	cvReleaseImage(&pv_big);
	
	secondsElapsed = difftime(timer_end,timer_begin);
	
	printf ("%.f seconds for %d frames : FPS = %f\n", secondsElapsed,nCount,(float)((float)(nCount)/secondsElapsed));
		
	return 0;
}
