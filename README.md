# 2025-RMV-task2
文件目录：

opencv_project \
├─ results\
│  ├─ watershed_bg_fill.png           //漫水处理-填充\
│  ├─ gray_image.png                  //灰度图\
│  ├─ watershed_boundaries.png        //漫水处理-分割\
│  ├─ red_contours.png                //红色轮廓\
│  ├─ cropped_quarter.png             //四分之一\
│  ├─ red_mask.png                    //红色区域\
│  ├─ morphology.png                  //膨胀+腐蚀\
│  ├─ rotated_35.png                  //旋转35度\
│  ├─ hsv_image.png                   //hsv\
│  ├─ adaptive_threshold.png          //二值化\
│  ├─ mean_blur.png                   //均值滤波\
│  ├─ drawing.png                     //绘图\
│  ├─ 程序运行终端截图.png        
│  ├─ gauss_blur.png                  //高斯滤波\
│  ├─ number_recognition.png          //数字框选+识别\
│  └─ armor_recognition.png           //灯条框选\
├─ CMakeLists.txt\
├─ src\
│  └─ main.cpp\
└─ resources\
   ├─ test_image.png\
   └─ test_image_2.png

   调试日志\
   1.一开始发现背景和花朵的分割很不明显，检查每一步输出结果后发现，应当先使用膨胀后使用腐蚀，消除背景中的黑点。\
   2.在实现灯条的框选时，发现没办法识别出灯条，推测可能是inRange函数出现了问题，于是写了另一个程序，用于实时测定图片的HSV，结果发现灯条的饱和度特别低，于是调低阈值，遂通过。\
   3.在实现数字的框选时，本来使用颜色边缘检测，发现效果很差，换成亮度边缘检测之后结果变好。
