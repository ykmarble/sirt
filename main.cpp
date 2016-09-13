#define EIGEN_NO_DEBUG

#include <eigen3/Eigen/Core>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <memory>

namespace {

const int NumOfAngle = 512;

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
    img_t *= 255;
    unsigned int width = img_t.cols();
    unsigned int height = img_t.rows();
    FILE* f = fopen(path, "wb");
    fwrite(magic, sizeof(char), 4, f);
    fwrite(&width, sizeof(unsigned int), 1, f);
    fwrite(&height, sizeof(unsigned int), 1, f);
    fwrite(&img_t(0), sizeof(float), width * height, f);
    fclose(f);
}

void inner_proj(MatrixXf *img, MatrixXf *proj, bool inverse) {
    // 画像の中心(0, 0)から下ろした垂線は角度によらず*projのド真ん中に落ちることをうまく使う
    // とりあえずは平行ビームで考えると、pixelと(0, 0)を通る垂線との距離がそのまま*projの中心との距離になる
    // ファンビームではX線源の位置も関わってくるが、X線源を中心とした同心円の中心を通る垂線との弧(あるいはその三角形)を考えて相似を使えばいいと思われる
    // Pointのk(cos t, sin t)Tへの射影
    // *projの長さは高々画像の対角線である
    float width_offset = (img->cols() - 1) / 2.0;
    float height_offset = (img->rows() - 1) / 2.0;
    int detector_centor = proj->rows() / 2;
    for (int deg_i = 0; deg_i < NumOfAngle; deg_i++) {
        float deg = (float)deg_i / NumOfAngle * 2 * M_PI;
        float a = sin(deg);
        float b = -cos(deg);
        for (int y_i = 0; y_i < img->rows(); y_i++) {
            for (int x_i = 0; x_i < img->cols(); x_i++) {
                float x = x_i - width_offset;
                float y  = (img->rows() - 1 - y_i) - height_offset;
                int sign = b * y > a * x ? 1 : -1;
                float dist = sign * abs(a * x + b * y);
                int l = floor(dist) + detector_centor;
                int h = l + 1;
                float h_ratio = dist - floor(dist);
                float l_ratio = 1 - h_ratio;
                if (inverse) {
                    (*img)(y_i, x_i) += (*proj)(l, deg_i) * l_ratio / NumOfAngle;
                    (*img)(y_i, x_i) += (*proj)(h, deg_i) * h_ratio / NumOfAngle;
                } else {
                    float val = (*img)(y_i, x_i) / (2 * proj->rows());
                    (*proj)(l, deg_i) += val * l_ratio;
                    (*proj)(h, deg_i) += val * h_ratio;
                }
            }
        }
    }
}

void projection(const MatrixXf &img ,MatrixXf *proj) {
    inner_proj((MatrixXf*)&img, proj, false);
}

void inv_projection(const MatrixXf &proj, MatrixXf *img) {
    inner_proj(img, (MatrixXf*)&proj, true);
}

void sirt(const MatrixXf &data, MatrixXf *img) {
    *img = MatrixXf::Zero(img->rows(), img->cols());
    inv_projection(data, img);
    MatrixXf proj = MatrixXf::Zero(data.rows(), data.cols());;
    MatrixXf grad = MatrixXf::Zero(img->rows(), img->cols());;
    float alpha = 5;
    int i = 0;
    float error = 10000;
    projection(*img, &proj);
    while (error > 0.1 ) {
        grad = MatrixXf::Zero(img->rows(), img->cols());
        inv_projection(data - proj, &grad);
        *img += alpha * grad;
        proj = MatrixXf::Zero(data.rows(), data.cols());
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
    save_rawimage("out/proj.dat", proj);
    printf("project\n");
    MatrixXf inv_proj = MatrixXf::Zero(img.rows(), img.cols());
    inv_projection(proj, &inv_proj);
    save_rawimage("out/inv_proj.dat", inv_proj);
    printf("inv project\n");
    MatrixXf recon = MatrixXf::Zero(img.rows(), img.cols());
    sirt(proj, &recon);
    save_rawimage("out/recon.dat", recon);
    return 0;
}
