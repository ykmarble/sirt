#define EIGEN_NO_DEBUG

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
    img_t = (img_t.array() - img_t.minCoeff()).matrix();
    img_t /= img_t.maxCoeff();
    unsigned int width = img_t.cols();
    unsigned int height = img_t.rows();
    FILE* f = fopen(path, "wb");
    fwrite(magic, sizeof(char), 4, f);
    fwrite(&width, sizeof(unsigned int), 1, f);
    fwrite(&height, sizeof(unsigned int), 1, f);
    fwrite(&img_t(0), sizeof(float), width * height, f);
    fclose(f);
}


void projection(const MatrixXf &img ,MatrixXf *proj) {
    // 画像の中心(0, 0)から下ろした垂線は角度によらず*projのド真ん中に落ちることをうまく使う
    // とりあえずは平行ビームで考えると、pixelと(0, 0)を通る垂線との距離がそのまま*projの中心との距離になる
    // ファンビームではX線源の位置も関わってくるが、X線源を中心とした同心円の中心を通る垂線との弧(あるいはその三角形)を考えて相似を使えばいいと思われる
    // Pointのk(cos t, sin t)Tへの射影
    // *projの長さは高々画像の対角線である
    float width_2 = img.cols() / 2.0;
    float height_2 = img.rows() / 2.0;
    int detector_offset = proj->rows() / 2;
    float val;
    Vector2f p;  // normalized position
    Vector2f base;
    for (int deg_i = 0; deg_i < NumOfAngle; deg_i++) {
        float deg = (float)deg_i / NumOfAngle * 2 * M_PI;
        float a = tan(deg);
        int right_hand_sign = deg > M_PI_2 && deg < 3 * M_PI_2 ? -1 : 1;
        base << cos(deg), sin(deg);
        for (int y = 0; y < img.rows(); y++) {
            for (int x = 0; x < img.cols(); x++) {
                val = img(y, x);
                p(0) = (x - width_2) / width_2;
                p(1) = - (y - height_2) / height_2;
                int is_higher = p(1) > a * p(0) ? 1 : -1;
                int sign = is_higher * right_hand_sign;
                float dist = (p - p.dot(base) * base).norm() * height_2;
                dist *= sign;
                int l = floor(dist) + detector_offset;
                int h = l + 1;
                float h_ratio = dist - floor(dist);
                float l_ratio = 1 - h_ratio;
                (*proj)(l, deg_i) += val * l_ratio;
                (*proj)(h, deg_i) += val * h_ratio;
            }
        }
    }
    float denom = max(-proj->minCoeff(), proj->maxCoeff());
    if (denom != 0)
        *proj /= denom;
    printf("proj: %f %f %f %f\n", img.minCoeff(), img.maxCoeff(), proj->minCoeff(), proj->maxCoeff());
}

void inv_projection(const MatrixXf &proj, MatrixXf *img) {
    float width_2 = img->cols() / 2.0;
    float height_2 = img->rows() / 2.0;
    int detector_offset = proj.rows() / 2;
    Vector2f p;  // normalized position
    Vector2f base;
    for (int deg_i = 0; deg_i < NumOfAngle; deg_i++) {
        float deg = (float)deg_i / NumOfAngle * 2 * M_PI;
        float a = tan(deg);
        int right_hand_sign = deg > M_PI_2 && deg < 3 * M_PI_2 ? -1 : 1;
        base << cos(deg), sin(deg);
        for (int y = 0; y < img->rows(); y++) {
            for (int x = 0; x < img->cols(); x++) {
                p(0) = (x - width_2) / width_2;
                p(1) = - (y - height_2) / height_2;
                int is_higher = p(1) > a * p(0) ? 1 : -1;
                int sign = is_higher * right_hand_sign;
                float dist = (p - p.dot(base) * base).norm() * height_2;
                dist *= sign;
                int l = floor(dist) + detector_offset;
                int h = l + 1;
                float h_ratio = dist - floor(dist);
                float l_ratio = 1 - h_ratio;
                (*img)(y, x) += proj(l, deg_i) * l_ratio + proj(h, deg_i) * h_ratio;
            }
        }
    }
    float denom = max(-img->minCoeff(), img->maxCoeff());
    if (denom != 0)
        *img /= denom;
    printf("inv proj: %f %f %f %f\n", proj.minCoeff(), proj.maxCoeff(), img->minCoeff(), img->maxCoeff());
}

void sirt(const MatrixXf &data, MatrixXf *img) {
    *img = MatrixXf::Zero(img->rows(), img->cols());
    MatrixXf proj = MatrixXf::Zero(data.rows(), data.cols());
    MatrixXf grad = MatrixXf::Zero(img->rows(), img->cols());
    float alpha = 0.000005;
    int i = 0;
    float error = 1;
    float last_error = error;
    projection(*img, &proj);
    while (error <= last_error) {
        last_error = error;
        inv_projection(data - proj, &grad);
        *img += alpha * grad;
        projection(*img, &proj);
        error = (proj - data).array().abs().sum() / (data.cols() * data.rows());
        i++;
        printf("%d: %f\n", i, error);
    }
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
    sirt(proj, &inv_proj);
    save_rawimage("out_sirt.dat", inv_proj);
    return 0;
}
