# -- coding: utf-8 --

import sys
from ctypes import *
import datetime
import numpy as np
import cv2
import traceback

sys.path.append("MVSDK")
from IMVApi import *

class Camera:
    def __init__(self):
        self.cam = None
        self.is_opened = False
        
    def _display_device_info(self, deviceInfoList):
        """Display available cameras"""
        print("Idx  Type   Vendor              Model           S/N                 DeviceUserID    IP Address")
        print("------------------------------------------------------------------------------------------------")
        for i in range(0, deviceInfoList.nDevNum):
            pDeviceInfo = deviceInfoList.pDevInfo[i]
            strType = ""
            try:
                strVendorName = pDeviceInfo.vendorName.decode("ascii")
                strModeName = pDeviceInfo.modelName.decode("ascii")
                strSerialNumber = pDeviceInfo.serialNumber.decode("ascii")
                strCameraname = pDeviceInfo.cameraName.decode("ascii")
                strIpAdress = pDeviceInfo.DeviceSpecificInfo.gigeDeviceInfo.ipAddress.decode("ascii")
            except Exception:
                strVendorName = str(pDeviceInfo.vendorName)
                strModeName = str(pDeviceInfo.modelName)
                strSerialNumber = str(pDeviceInfo.serialNumber)
                strCameraname = str(pDeviceInfo.cameraName)
                strIpAdress = str(pDeviceInfo.DeviceSpecificInfo.gigeDeviceInfo.ipAddress)

            if pDeviceInfo.nCameraType == typeGigeCamera:
                strType = "Gige"
            elif pDeviceInfo.nCameraType == typeU3vCamera:
                strType = "U3V"
            print ("[%d]  %s   %s    %s      %s     %s           %s" % (i+1, strType, strVendorName, strModeName, strSerialNumber, strCameraname, strIpAdress))

    def open(self, camera_index=None):
        """Open camera connection"""
        try:
            deviceList = IMV_DeviceList()
            interfaceType = IMV_EInterfaceType.interfaceTypeAll

            # Enumerate devices
            nRet = MvCamera.IMV_EnumDevices(deviceList, interfaceType)
            if IMV_OK != nRet:
                print("Enumeration devices failed! ErrorCode", nRet)
                return False
                
            if deviceList.nDevNum == 0:
                print("No device found!")
                return False

            print("Found", deviceList.nDevNum, "device(s)")
            self._display_device_info(deviceList)

            # Get camera index from user if not provided
            if camera_index is None:
                nConnectionNum = input("Please input the camera index: ")
                try:
                    camera_index = int(nConnectionNum)
                    if camera_index > deviceList.nDevNum or camera_index < 1:
                        print("Input error!")
                        return False
                except Exception:
                    print("Invalid input")
                    return False
            
            self.cam = MvCamera()
            
            # Create device handle (SDK uses 0-based index)
            nRet = self.cam.IMV_CreateHandle(IMV_ECreateHandleMode.modeByIndex, byref(c_void_p(camera_index-1)))
            if IMV_OK != nRet:
                print("Create devHandle failed! ErrorCode", nRet)
                return False

            # Open camera
            nRet = self.cam.IMV_Open()
            if IMV_OK != nRet:
                print("Open devHandle failed! ErrorCode", nRet)
                return False

            # Set trigger mode settings
            nRet = self.cam.IMV_SetEnumFeatureSymbol("TriggerSource", "Software")
            if IMV_OK != nRet:
                print("Set triggerSource value failed! ErrorCode[%d]" % nRet)
                return False

            nRet = self.cam.IMV_SetEnumFeatureSymbol("TriggerSelector", "FrameStart")
            if IMV_OK != nRet:
                print("Set triggerSelector value failed! ErrorCode[%d]" % nRet)
                return False

            nRet = self.cam.IMV_SetEnumFeatureSymbol("TriggerMode", "Off")
            if IMV_OK != nRet:
                print("Set triggerMode value failed! ErrorCode[%d]" % nRet)
                return False

            # Start grabbing
            nRet = self.cam.IMV_StartGrabbing()
            if IMV_OK != nRet:
                print("Start grabbing failed! ErrorCode", nRet)
                return False

            self.is_opened = True
            return True

        except Exception as e:
            print("Exception in open():", e)
            traceback.print_exc()
            return False

    def get_frame(self, timeout_ms=1000):
        """Get a single frame from camera"""
        if not self.is_opened or self.cam is None:
            print("Camera not opened!")
            return None

        try:
            frame = IMV_Frame()
            nRet = self.cam.IMV_GetFrame(frame, timeout_ms)
            
            if IMV_OK != nRet:
                print("getFrame fail! Timeout:[%d]ms" % timeout_ms)
                return None

            # Check if pointer is valid
            if not bool(frame.pData):
                print("pFrame is NULL or invalid pointer!")
                return None

            # Prepare pixel conversion parameters
            stPixelConvertParam = IMV_PixelConvertParam()
            memset(byref(stPixelConvertParam), 0, sizeof(stPixelConvertParam))
            stPixelConvertParam.nWidth = frame.frameInfo.width
            stPixelConvertParam.nHeight = frame.frameInfo.height
            stPixelConvertParam.ePixelFormat = frame.frameInfo.pixelFormat
            stPixelConvertParam.pSrcData = frame.pData
            stPixelConvertParam.nSrcDataLen = frame.frameInfo.size
            stPixelConvertParam.nPaddingX = frame.frameInfo.paddingX
            stPixelConvertParam.nPaddingY = frame.frameInfo.paddingY
            stPixelConvertParam.eBayerDemosaic = IMV_EBayerDemosaic.demosaicNearestNeighbor

            cvImage = None

            try:
                # Handle Mono8 format
                if frame.frameInfo.pixelFormat == IMV_EPixelType.gvspPixelMono8:
                    nBytes = int(frame.frameInfo.width * frame.frameInfo.height)
                    userBuff = (c_ubyte * nBytes)()
                    
                    # Copy data from driver buffer to user buffer
                    memmove(userBuff, frame.pData, nBytes)
                    
                    # Create numpy array
                    arr = np.frombuffer(userBuff, dtype=np.uint8, count=nBytes)
                    cvImage = arr.reshape((frame.frameInfo.height, frame.frameInfo.width))

                else:
                    # Handle color formats - convert to BGR24
                    nDstBufSize = int(frame.frameInfo.width * frame.frameInfo.height * 3)
                    pDstBuf = (c_ubyte * nDstBufSize)()

                    stPixelConvertParam.eDstPixelFormat = IMV_EPixelType.gvspPixelBGR8
                    stPixelConvertParam.pDstBuf = pDstBuf
                    stPixelConvertParam.nDstBufSize = nDstBufSize

                    nRet = self.cam.IMV_PixelConvert(stPixelConvertParam)
                    if IMV_OK != nRet:
                        print("PixelConvert failed! ErrorCode[%d]" % nRet)
                        return None

                    # Copy to numpy array
                    rgbBuff = (c_ubyte * stPixelConvertParam.nDstBufSize)()
                    memmove(rgbBuff, stPixelConvertParam.pDstBuf, stPixelConvertParam.nDstBufSize)
                    arr = np.frombuffer(rgbBuff, dtype=np.uint8, count=stPixelConvertParam.nDstBufSize)
                    cvImage = arr.reshape((stPixelConvertParam.nHeight, stPixelConvertParam.nWidth, 3))

                # Release frame after processing
                nRet = self.cam.IMV_ReleaseFrame(frame)
                if IMV_OK != nRet:
                    print("Release frame failed! ErrorCode[%d]" % nRet)

                return cvImage

            except Exception as e:
                # Release frame on exception
                try:
                    self.cam.IMV_ReleaseFrame(frame)
                except Exception:
                    pass
                print("Exception while processing frame:", e)
                traceback.print_exc()
                return None

        except Exception as e:
            print("Exception in get_frame():", e)
            traceback.print_exc()
            return None

    def close(self):
        """Close camera and cleanup resources"""
        if not self.is_opened or self.cam is None:
            return

        try:
            # Stop grabbing
            nRet = self.cam.IMV_StopGrabbing()
            if IMV_OK != nRet:
                print("Stop grabbing failed! ErrorCode", nRet)
        except Exception:
            pass

        try:
            # Close camera
            nRet = self.cam.IMV_Close()
            if IMV_OK != nRet:
                print("Close camera failed! ErrorCode", nRet)
        except Exception:
            pass

        try:
            # Destroy handle
            if self.cam.handle:
                nRet = self.cam.IMV_DestroyHandle()
        except Exception:
            pass

        self.is_opened = False
        self.cam = None
        print("Camera closed successfully")

    def __del__(self):
        """Destructor to ensure cleanup"""
        if self.is_opened:
            self.close()


