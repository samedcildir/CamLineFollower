cmake_minimum_required(VERSION 2.8)
project( CamLineFollower )
SET(COMPILE_DEFINITIONS -Werror)

#OPENCV
find_package( OpenCV REQUIRED )

set (GL_SCENE_SOURCES
gl_scenes/models.c
gl_scenes/mirror.c
gl_scenes/yuv.c
gl_scenes/sobel.c
gl_scenes/square.c
gl_scenes/teapot.c)
SET(CMAKE_EXE_LINKER_FLAGS "-lpthread")
include_directories(/home/pi/CamLineFollower/gl_scenes)
include_directories(/opt/vc/include)
include_directories(/opt/vc/include/EGL)
include_directories(/opt/vc/userland-master/host_applications/linux/libs/bcm_host/include)
include_directories(/opt/vc/userland/interface/vcos)
include_directories(/opt/vc/userland)
include_directories(/opt/vc/userland/interface/vcos/pthreads)
include_directories(/opt/vc/userland/interface/vmcs_host/linux)
include_directories(/home/pi/CamLineFollower)
add_executable(CamLineFollowerRaspiCamControl.c RaspiCLI.c RaspiPreview.c 
GPIOClass.cpp 
CamLineFollower.cpp RaspiTex.c RaspiTexUtil.c tga.c ${GL_SCENE_SOURCES})
target_link_libraries(CamLineFollowerm /opt/vc/lib/libmmal_core.so /opt/vc/lib/libmmal_util.so /opt/vc/lib/libmmal_vc_client.so /opt/vc/lib/libvcos.so /opt/vc/lib/libbcm_host.so /opt/vc/lib/libGLESv2.so /opt/vc/lib/libEGL.so /home/pi/libs/libfacerec/libopencv_facerec.a wiringPi pthread ${OpenCV_LIBS})
