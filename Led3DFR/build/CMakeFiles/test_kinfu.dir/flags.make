# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# compile CXX with /usr/bin/c++
CXX_FLAGS =   -std=gnu++14

CXX_DEFINES = 

CXX_INCLUDES = -isystem /data/irip_rpt/3rdparty/opencv-4.0.0/build -isystem /data/irip_rpt/3rdparty/opencv-4.0.0/include -isystem /data/irip_rpt/3rdparty/opencv_contrib-master/modules/cudev/include -isystem /data/irip_rpt/3rdparty/opencv-4.0.0/modules/core/include -isystem /data/irip_rpt/3rdparty/opencv_contrib-master/modules/cudaarithm/include -isystem /data/irip_rpt/3rdparty/opencv-4.0.0/modules/flann/include -isystem /data/irip_rpt/3rdparty/opencv_contrib-master/modules/hdf/include -isystem /data/irip_rpt/3rdparty/opencv-4.0.0/modules/imgproc/include -isystem /data/irip_rpt/3rdparty/opencv-4.0.0/modules/ml/include -isystem /data/irip_rpt/3rdparty/opencv_contrib-master/modules/phase_unwrapping/include -isystem /data/irip_rpt/3rdparty/opencv_contrib-master/modules/plot/include -isystem /data/irip_rpt/3rdparty/opencv_contrib-master/modules/reg/include -isystem /data/irip_rpt/3rdparty/opencv_contrib-master/modules/surface_matching/include -isystem /data/irip_rpt/3rdparty/opencv_contrib-master/modules/viz/include -isystem /data/irip_rpt/3rdparty/opencv_contrib-master/modules/cudafilters/include -isystem /data/irip_rpt/3rdparty/opencv_contrib-master/modules/cudaimgproc/include -isystem /data/irip_rpt/3rdparty/opencv_contrib-master/modules/cudawarping/include -isystem /data/irip_rpt/3rdparty/opencv-4.0.0/modules/dnn/include -isystem /data/irip_rpt/3rdparty/opencv_contrib-master/modules/freetype/include -isystem /data/irip_rpt/3rdparty/opencv_contrib-master/modules/fuzzy/include -isystem /data/irip_rpt/3rdparty/opencv-4.0.0/modules/gapi/include -isystem /data/irip_rpt/3rdparty/opencv_contrib-master/modules/hfs/include -isystem /data/irip_rpt/3rdparty/opencv_contrib-master/modules/img_hash/include -isystem /data/irip_rpt/3rdparty/opencv-4.0.0/modules/imgcodecs/include -isystem /data/irip_rpt/3rdparty/opencv-4.0.0/modules/photo/include -isystem /data/irip_rpt/3rdparty/opencv-4.0.0/modules/videoio/include -isystem /data/irip_rpt/3rdparty/opencv_contrib-master/modules/xphoto/include -isystem /data/irip_rpt/3rdparty/opencv_contrib-master/modules/cudacodec/include -isystem /data/irip_rpt/3rdparty/opencv-4.0.0/modules/highgui/include -isystem /data/irip_rpt/3rdparty/opencv-4.0.0/modules/ts/include -isystem /data/irip_rpt/3rdparty/opencv_contrib-master/modules/bioinspired/include -isystem /data/irip_rpt/3rdparty/opencv_contrib-master/modules/dnn_objdetect/include -isystem /data/irip_rpt/3rdparty/opencv-4.0.0/modules/features2d/include -isystem /data/irip_rpt/3rdparty/opencv_contrib-master/modules/line_descriptor/include -isystem /data/irip_rpt/3rdparty/opencv_contrib-master/modules/saliency/include -isystem /data/irip_rpt/3rdparty/opencv_contrib-master/modules/text/include -isystem /data/irip_rpt/3rdparty/opencv-4.0.0/modules/calib3d/include -isystem /data/irip_rpt/3rdparty/opencv_contrib-master/modules/ccalib/include -isystem /data/irip_rpt/3rdparty/opencv_contrib-master/modules/cudafeatures2d/include -isystem /data/irip_rpt/3rdparty/opencv_contrib-master/modules/cudastereo/include -isystem /data/irip_rpt/3rdparty/opencv_contrib-master/modules/datasets/include -isystem /data/irip_rpt/3rdparty/opencv-4.0.0/modules/objdetect/include -isystem /data/irip_rpt/3rdparty/opencv_contrib-master/modules/rgbd/include -isystem /data/irip_rpt/3rdparty/opencv_contrib-master/modules/shape/include -isystem /data/irip_rpt/3rdparty/opencv_contrib-master/modules/stereo/include -isystem /data/irip_rpt/3rdparty/opencv_contrib-master/modules/structured_light/include -isystem /data/irip_rpt/3rdparty/opencv-4.0.0/modules/video/include -isystem /data/irip_rpt/3rdparty/opencv_contrib-master/modules/xfeatures2d/include -isystem /data/irip_rpt/3rdparty/opencv_contrib-master/modules/ximgproc/include -isystem /data/irip_rpt/3rdparty/opencv_contrib-master/modules/xobjdetect/include -isystem /data/irip_rpt/3rdparty/opencv_contrib-master/modules/aruco/include -isystem /data/irip_rpt/3rdparty/opencv_contrib-master/modules/bgsegm/include -isystem /data/irip_rpt/3rdparty/opencv_contrib-master/modules/cudabgsegm/include -isystem /data/irip_rpt/3rdparty/opencv_contrib-master/modules/cudalegacy/include -isystem /data/irip_rpt/3rdparty/opencv_contrib-master/modules/cudaobjdetect/include -isystem /data/irip_rpt/3rdparty/opencv_contrib-master/modules/dpm/include -isystem /data/irip_rpt/3rdparty/opencv_contrib-master/modules/face/include -isystem /data/irip_rpt/3rdparty/opencv_contrib-master/modules/optflow/include -isystem /data/irip_rpt/3rdparty/opencv_contrib-master/modules/sfm/include -isystem /data/irip_rpt/3rdparty/opencv-4.0.0/modules/stitching/include -isystem /data/irip_rpt/3rdparty/opencv_contrib-master/modules/tracking/include -isystem /data/irip_rpt/3rdparty/opencv_contrib-master/modules/cudaoptflow/include -isystem /data/irip_rpt/3rdparty/opencv_contrib-master/modules/superres/include -isystem /data/irip_rpt/3rdparty/opencv_contrib-master/modules/videostab/include -I/data/irip_rpt/3rdparty/mxnet/include -I/data/irip_rpt/3DFR/Led3DFR/include/Led3DFR 

