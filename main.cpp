#include <eigen3/Eigen/Core>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <memory>

namespace {

const int NumOfAngle = 360;

using namespace Eigen;
using namespace std;

MatrixXf load_rawimage(const char *path) {
    FILE* f = fopen(path, "rb");
    char magic[4];
    unsigned int width, height;
    fread(magic, sizeof(char), 4, f);
    fread(&width, sizeof(unsigned int), 1, f);
    fread(&height, sizeof(unsigned int), 1, f);
    unique_ptr<float> img_seq(new float[width * height]);
    fread(img_seq.get(), sizeof(float), width * height, f);
    fclose(f);
    return Eigen::Map<Matrix<float, Dynamic, Dynamic, RowMajor>> (img_seq.get(), width, height);;
}

void save_rawimage(const char *path, const MatrixXf &img) {
    char magic[] = {'P', '0', 0x00, 0x00};
    Matrix<float, Dynamic, Dynamic, RowMajor> img_t = img;
    unsigned int width = img_t.cols();
    unsigned int height = img_t.rows();
    FILE* f = fopen(path, "wb");
    fwrite(magic, sizeof(char), 4, f);
    fwrite(&width, sizeof(unsigned int), 1, f);
    fwrite(&height, sizeof(unsigned int), 1, f);
    fwrite(&img_t(0), sizeof(float), width * height, f);
    fclose(f);
}

void index_in_Point(int x, int y, int width, int height, Vector2f* p) {
    (*p)(0) = (x - width / 2.0) / (width / 2.0);
    (*p)(1) = - (y - height / 2.0) / (height / 2.0);
}

float index_in_deg(int index) {
    return (float)index / NumOfAngle * 2 * M_PI;
}

int calc_proj_sign(const Vector2f &pixel, const float deg) {
    float y = pixel(1);
    float x = pixel(0);
    float a = tan(deg);
    int right_hand_sign = deg > M_PI_2 && deg < 3 * M_PI_2 ? -1 : 1;
    int is_higher = y > a * x ? 1 : -1;
    return is_higher * right_hand_sign;
}

float point_on_detector(const Vector2f &pixel, float deg) {
    Vector2f base(cos(deg), sin(deg));
    float r = pixel.dot(base);
    float dist = (pixel - r * base).norm();
    int sign = calc_proj_sign(pixel, deg);
    return sign * dist;
}

void projection(const MatrixXf &img ,MatrixXf *proj) {
    // 画像の中心(0, 0)から下ろした垂線は角度によらず*projのド真ん中に落ちることをうまく使う
    // とりあえずは平行ビームで考えると、pixelと(0, 0)を通る垂線との距離がそのまま*projの中心との距離になる
    // ファンビームではX線源の位置も関わってくるが、X線源を中心とした同心円の中心を通る垂線との弧(あるいはその三角形)を考えて相似を使えばいいと思われる
    // Pointのk(cos t, sin t)Tへの射影
    // *projの長さは高々画像の対角線である
    for (int y = 0; y < img.rows(); y++) {
        for (int x = 0; x < img.cols(); x++) {
            auto val = img(y, x);
            Vector2f p;
            index_in_Point(x, y, img.cols(), img.rows(), &p);
            for (int deg_i = 0; deg_i < NumOfAngle; deg_i++) {
                float deg = index_in_deg(deg_i);
                float dist = point_on_detector(p, deg) * img.rows() / 2.0;
                int l = floor(dist) + proj->rows() / 2;
                int h = l + 1;
                float h_ratio = dist - floor(dist);
                float l_ratio = 1 - h_ratio;
                (*proj)(l, deg_i) += val * l_ratio;
                (*proj)(h, deg_i) += val * h_ratio;
            }
        }
    }
    *proj /= proj->maxCoeff();
    *proj *= 255;
}

void inv_projection(const MatrixXf &proj, MatrixXf *img) {
    for (int y = 0; y < img->rows(); y++) {
        for (int x = 0; x < img->cols(); x++) {
            Vector2f p;
            index_in_Point(x, y, img->cols(), img->rows(), &p);
            for (int deg_i = 0; deg_i < NumOfAngle; deg_i++) {
                float deg = index_in_deg(deg_i);
                float dist = point_on_detector(p, deg) * img->rows() / 2.0;
                int l = floor(dist) + proj.rows() / 2;
                int h = l + 1;
                float h_ratio = dist - floor(dist);
                float l_ratio = 1 - h_ratio;
                (*img)(y, x) += proj(l, deg_i) * l_ratio + proj(h, deg_i) * h_ratio;
            }
        }
    }
    *img /= img->maxCoeff();
    *img *= 255;
}
}

int main(int argc, char** argv) {
    if (argc != 2) {
        printf("Usage %s file", argv[0]);
        return 1;
    }
    MatrixXf img = load_rawimage(argv[1]);
    int detector_len = ceil(sqrt(pow(img.rows(), 2) + pow(img.cols(), 2)));
    MatrixXf proj = MatrixXf::Zero(detector_len + 2, NumOfAngle);
    projection(img, &proj);
    save_rawimage("out_proj.dat", proj);
    printf("project\n");
    MatrixXf inv_proj = MatrixXf::Zero(img.rows(), img.cols());
    inv_projection(proj, &inv_proj);
    save_rawimage("out_inv_proj.dat", inv_proj);
    printf("inv project\n");
    return 0;
}
