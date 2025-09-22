#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>
#include <iostream>
#include <iomanip>

using namespace cv;
using namespace std;


int main() {
    // ---------- 读取图像 ----------
    string img_path = string(RESOURCE_DIR) + "/test_image.png";
    string img_path2 = string(RESOURCE_DIR) + "/test_image_2.png";
    Mat img = imread(img_path);
    Mat img2 = imread(img_path2);

    // ---------- 1. 颜色空间转换 ----------
    Mat gray, hsv;
    cvtColor(img, gray, COLOR_BGR2GRAY);   // 转灰度图
    cvtColor(img, hsv, COLOR_BGR2HSV);     // 转 HSV 图

    imshow("原图", img);
    imshow("灰度图", gray);
    imshow("HSV 图 ", hsv);

    //imwrite("results/gray_image.png", gray);
    //imwrite("/home/liuyuxi/my_cplus_project/opencv_project/results/hsv_image.png", hsv);


    // ---------- 2. 滤波操作 ----------
    Mat mean_blur, gauss_blur;
    blur(img, mean_blur, Size(5,5));                // 均值滤波
    GaussianBlur(img, gauss_blur, Size(5,5), 1.5);  // 高斯滤波
    imshow("均值滤波", mean_blur);
    imshow("高斯滤波", gauss_blur);

    //imwrite("/home/liuyuxi/my_cplus_project/opencv_project/results/mean_blur.png", mean_blur);
    //imwrite("/home/liuyuxi/my_cplus_project/opencv_project/results/gauss_blur.png", gauss_blur);

    // ---------- 3. 提取红色区域（HSV 方法） ----------
    Mat mask1, mask2, red_mask;
    inRange(hsv, Scalar(0, 100, 100), Scalar(10, 255, 255), mask1);
    inRange(hsv, Scalar(160, 100, 100), Scalar(180, 255, 255), mask2);
    red_mask = mask1 | mask2;

    imshow("红色区域", red_mask);

    //imwrite("/home/liuyuxi/my_cplus_project/opencv_project/results/red_mask.png", red_mask);

    // ---------- 4. 寻找外轮廓、bounding box、计算面积，并绘制（红色） ----------
    vector<vector<Point>> contours;
    findContours(red_mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    Mat contour_img = img.clone();
    double total_red_area = 0.0;
    const double AREA_THRESHOLD = 50.0; // 忽略极小噪声

    for (size_t i = 0; i < contours.size(); ++i) {
        double area = fabs(contourArea(contours[i]));
        if (area < AREA_THRESHOLD) continue;
        total_red_area += area;

        // 绘制外轮廓（红色）
        drawContours(contour_img, contours, (int)i, Scalar(0,0,255), 2);

        // 计算并绘制 bounding box（红色）
        Rect box = boundingRect(contours[i]);
        rectangle(contour_img, box, Scalar(0,0,255), 2);

        // 在 bounding box 上方写面积（像素）
        string txt = to_string((int)area) + " px";
        int baseline = 0;
        double fontScale = 0.5;
        int thickness = 1;
        Size txtSize = getTextSize(txt, FONT_HERSHEY_SIMPLEX, fontScale, thickness, &baseline);
        Point textOrg(box.x, max(0, box.y - 5));
        putText(contour_img, txt, textOrg, FONT_HERSHEY_SIMPLEX, fontScale, Scalar(0,255,0), thickness);
    }

    double img_area = double(img.cols) * img.rows;
    double red_percent = img_area > 0.0 ? (total_red_area / img_area * 100.0) : 0.0;
    cout << "轮廓数 (raw): " << contours.size() << endl;
    cout << "红色总面积: " << (int)total_red_area << " px, 占比: " << fixed << setprecision(2) << red_percent << "%" << endl;

    imshow("红色区域轮廓与 bounding box (红色绘制)", contour_img);

    //imwrite("/home/liuyuxi/my_cplus_project/opencv_project/results/red_contours.png", contour_img);

    // ---------- 5. 提取高亮区域并进行形态学处理（灰度->二值->闭运算） ----------

    Mat bin_img;
    adaptiveThreshold(gray, bin_img, 
                  255, 
                  ADAPTIVE_THRESH_GAUSSIAN_C, 
                  THRESH_BINARY, 
                  11, 2);

    imshow("自适应阈值二值化", bin_img);

    //imwrite("/home/liuyuxi/my_cplus_project/opencv_project/results/adaptive_threshold.png", bin_img);

    // 闭运算
    Mat dilateimg ,erodeimg;
    Mat kernel = getStructuringElement(MORPH_RECT, Size(2,2));
    dilate(bin_img, dilateimg, kernel);
    erode(dilateimg, erodeimg, kernel);

    imshow("先膨胀后腐蚀", erodeimg);

    //imwrite("/home/liuyuxi/my_cplus_project/opencv_project/results/morphology.png", erodeimg);


    // ---------- 6. 对处理后的图像进行漫水（watershed）分割 ----------
    // 准备 markers：用距离变换确定前景、膨胀确定背景
    Mat dist;
    distanceTransform(erodeimg, dist, DIST_L2, 5);
    normalize(dist, dist, 0, 1.0, NORM_MINMAX);

    Mat sure_fg;
    threshold(dist, sure_fg, 0.8, 1.0, THRESH_BINARY); // 0.4 可调
    // 转回 8U
    Mat sure_fg_8u;
    sure_fg.convertTo(sure_fg_8u, CV_8U, 255);

    Mat sure_bg;
    dilate(erodeimg, sure_bg, kernel, Point(-1,-1), 3); // 大范围膨胀作为背景

    Mat unknown;
    subtract(sure_bg, sure_fg_8u, unknown);

    // 连通组件作为 markers
    Mat markers;
    int nComps = connectedComponents(sure_fg_8u, markers);
    markers = markers + 1;
    // 把未知区域设为 0
    for (int r = 0; r < unknown.rows; ++r) {
        for (int c = 0; c < unknown.cols; ++c) {
            if (unknown.at<uchar>(r,c) == 255) {
                markers.at<int>(r,c) = 0;
            }
        }
    }

    // watershed 要求输入为 3 通道 BGR
    Mat img_for_watershed;
    img.copyTo(img_for_watershed);
    watershed(img_for_watershed, markers);

    // 将分水岭边界 (marker == -1) 标为红色
    Mat wshed_vis = img.clone();
    for (int r = 0; r < markers.rows; ++r) {
        for (int c = 0; c < markers.cols; ++c) {
            if (markers.at<int>(r,c) == -1) {
                wshed_vis.at<Vec3b>(r,c) = Vec3b(0,0,255); // 红色线
            }
        }
    }
    imshow("Watershed 分割结果 (边界为红色)", wshed_vis);

    //imwrite("/home/liuyuxi/my_cplus_project/opencv_project/results/watershed_boundaries.png", wshed_vis);

     Mat bg_fill = img.clone();

    for (int r = 0; r < markers.rows; r++) {
    for (int c = 0; c < markers.cols; c++) {
        int label = markers.at<int>(r,c);
        if (label >1) { 
            // 背景
            bg_fill.at<Vec3b>(r,c) = Vec3b(0,255,0); // 绿色
        }
        else if (label == -1) { 
            // 分水岭边界
            bg_fill.at<Vec3b>(r,c) = Vec3b(0,0,255); // 红色
        }
        // 其它 (label > 1) 前景，保持原图
    }
    }

     imshow("背景填充 ", bg_fill);

     //imwrite("/home/liuyuxi/my_cplus_project/opencv_project/results/watershed_bg_fill.png", bg_fill);



    // ---------- 7. 图像绘制（任意圆/方形/文字） ----------
    Mat draw_img = img.clone();
    // 圆
    circle(draw_img, Point(50,50), 30, Scalar(255,0,255), -1);
    // 矩形
    rectangle(draw_img, Rect(100,30,120,80), Scalar(255,255,0), 2);
    // 文字
    putText(draw_img, "Hello DXRMV", Point(100,150), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0,255,0), 2);
    imshow("绘制图形与文字", draw_img);

    //imwrite("/home/liuyuxi/my_cplus_project/opencv_project/results/drawing.png", draw_img);

    // ---------- 8. 图像旋转 35 度与左上角 1/4 裁剪 ----------
    Mat rot;
    Point2f center(img.cols/2.0F, img.rows/2.0F);
    Mat rotMat = getRotationMatrix2D(center, 35, 1.0);
    warpAffine(img, rot, rotMat, img.size());
    imshow("旋转 35 度", rot);
    //imwrite("/home/liuyuxi/my_cplus_project/opencv_project/results/rotated_35.png", rot);

    Rect roi(0, 0, img.cols/2, img.rows/2);
    Mat cropped = img(roi).clone();
    imshow("左上角 1/4 裁剪", cropped);
    //imwrite("/home/liuyuxi/my_cplus_project/opencv_project/results/cropped_quarter.png", cropped);


    // 转 HSV
Mat hsv2;
cvtColor(img2, hsv2, COLOR_BGR2HSV);


Mat mask3;
Scalar lower_blue(0, 1, 220);  
Scalar upper_blue(255, 100, 255);  
inRange(hsv2, lower_blue, upper_blue, mask3);

// 形态学操作：闭运算+开运算
Mat kernel2 = getStructuringElement(MORPH_RECT, Size(3, 3));
morphologyEx(mask3, mask3, MORPH_CLOSE, kernel2);
morphologyEx(mask3, mask3, MORPH_OPEN, kernel2);

// 找轮廓
vector<vector<Point>> contours2;
findContours(mask3, contours2, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

// 筛选并画矩形
for (auto &contour2 : contours2) {
    Rect box = boundingRect(contour2);
    float aspect = (float)box.height/ (float)box.width;

    // 过滤掉太小的干扰，并根据长宽比筛选灯条
    if (box.area() > 1000 &&box.area()<5000&&  aspect >2) {
        rectangle(img2, box, Scalar(0, 0, 255), 2);
    }
}

imshow("装甲板识别结果", img2);
//imwrite("/home/liuyuxi/my_cplus_project/opencv_project/results/armor_recognition.png", img2);


// 1. 转换到HSV
Mat hsv3;
cvtColor(img2, hsv3, COLOR_BGR2HSV);

// 2. 拆分通道
vector<Mat> hsv_channels(3);
split(hsv3, hsv_channels);
Mat H = hsv_channels[0];
Mat S = hsv_channels[1];
Mat V = hsv_channels[2];

// 3. 对V通道做Canny边缘检测
Mat edges;
Canny(V, edges, 200, 800);

// 4. 形态学操作闭合边缘
Mat kernel3 = getStructuringElement(MORPH_RECT, Size(3, 3));
morphologyEx(edges, edges, MORPH_CLOSE, kernel3);
dilate(edges, edges, kernel3, Point(-1,-1), 1);

// 5. 查找轮廓
vector<vector<Point>> contours3;
findContours(edges, contours3, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

// 6. 初始化 Tesseract OCR
tesseract::TessBaseAPI tess;
if (tess.Init("/usr/share/tesseract-ocr/4.00/tessdata", "eng") != 0) {
    cerr << "Tesseract 初始化失败！请检查 tessdata 路径是否正确" << endl;
    return -1; 
}
tess.SetPageSegMode(tesseract::PSM_SINGLE_CHAR);
tess.SetVariable("tessedit_char_whitelist", "0123456789"); // 只识别数字


// 7. 创建用于OCR的预处理图像
Mat gray4;
cvtColor(img2, gray4, COLOR_BGR2GRAY);

// 8. 遍历轮廓进行识别
for (auto &contour : contours3) {
    Rect box = boundingRect(contour);
    
    // 更严格的面积过滤
    if (box.area() < 1000 || box.area() > 50000) continue;
    
    // 宽高比过滤（数字通常有一定比例）
    double aspect_ratio = (double)box.width / box.height;
    if (aspect_ratio < 0.3 || aspect_ratio > 1.5) continue;
    
    // 扩展ROI区域，避免切割字符
    int padding = 5;
    Rect expanded_box = Rect(
        max(0, box.x - padding),
        max(0, box.y - padding),
        min(gray4.cols - box.x, box.width + 2 * padding),
        min(gray4.rows - box.y, box.height + 2 * padding)
    );
    
    Mat roi = gray4(expanded_box);
    
    // 图像预处理：二值化+降噪
    Mat processed_roi;
    threshold(roi, processed_roi, 0, 255, THRESH_BINARY | THRESH_OTSU);
    
    // 形态学操作去除噪声
    Mat kernel = getStructuringElement(MORPH_RECT, Size(2, 2));
    morphologyEx(processed_roi, processed_roi, MORPH_OPEN, kernel);
    
    // 确保数字是黑色，背景是白色（Tesseract偏好）

    bitwise_not(processed_roi, processed_roi);
    
    
    // 调整图像尺寸（Tesseract对大小敏感）
    Mat resized_roi;
    int target_size = 28;
    resize(processed_roi, resized_roi, Size(target_size, target_size));
    
    // OCR识别
    tess.SetImage(resized_roi.data, resized_roi.cols, resized_roi.rows, 1, resized_roi.step);
    char* outText = tess.GetUTF8Text();
    
    if (outText && strlen(outText) > 0 && isdigit(outText[0])) {
        rectangle(img2, expanded_box, Scalar(0, 255, 0), 2);
        putText(img2, string(1, outText[0]), Point(expanded_box.x, expanded_box.y - 5),
                FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 255), 2);
        cout << "识别到数字: " << outText[0] << " at " << expanded_box << endl;
    }
    
    delete[] outText;
}

// 9. 显示和保存结果
imshow("数字识别结果", img2);
//imwrite("/home/liuyuxi/my_cplus_project/opencv_project/results/number_recognition.png", img2);


    // ---------- 结束 ----------
    cout << "按ctrl+C关闭窗口..." << endl;
    waitKey(0);
    destroyAllWindows();
    return 0;
}
