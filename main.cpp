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
    /*
      `path`から独自形式の画像を読み込む。画素値の正規化は行わない。
     */
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
    /*
      `path`に`img`の画像を独自形式で書き出す。書き出す前に画素値の正規化を行う。
     */
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
    /*
      `inverse`がfalseの時、`img`を`proj`に順投影する。
      `inverse`がtrueの時、`proj`を`img`に逆投影する。
      投影はpallarel beamジオメトリでpixel-drivenに行われる。
      各画素から垂線を下ろした先に検知器が存在しない、つまり`proj`の長さが足りない場合エラーになる。
      `proj`は高々画像の対角線+2の長さがあれば足りる。
     */
    // memo:
    // 画素間の幅を1、画像の中心を(0, 0)として座標系を取る。
    // 検知器間の幅も1とし、角度0の時の検知器の中心のy座標を0としてy軸に平行に検知器が並ぶとする
    // 平行ビームなので角度0の時のx座標は、画像の対角線の半分以上であれば何でもいい。

    // 各indexに履かせる負の下駄の大きさ
    float width_offset = (img->cols() - 1) / 2.0;
    float height_offset = (img->rows() - 1) / 2.0;
    float detector_offset = (proj->rows() - 1) / 2.0;

    for (int deg_i = 0; deg_i < NumOfAngle; deg_i++) {
        float deg = (float)deg_i / NumOfAngle * 2 * M_PI;

        // 原点を通る検知器列に直行する直線 ax + yb = 0
        float a = sin(deg);
        float b = -cos(deg);
        for (int y_i = 0; y_i < img->rows(); y_i++) {
            for (int x_i = 0; x_i < img->cols(); x_i++) {
                float x = x_i - width_offset;
                float y  = height_offset - y_i;  // 行列表記と軸の方向が逆になることに注意

                // distは検知器中心からの距離を意味しているが、X線源に向かって右側を正とする
                // ローカル座標で表されている。
                int sign = b * y > a * x ? 1 : -1;  // 角度によらずこれで符号が出る
                float dist = sign * abs(a * x + b * y);

                // 影響のある検知器は高々2つ、検知器上のローカル座標系でより小さい方をlとする
                int l = floor(dist + detector_offset);
                int h = l + 1;
                float h_ratio = dist - floor(dist);
                float l_ratio = 1 - h_ratio;

                // 共に正規化の仕方が雑で雑にしか検証していない。
                // 入力の最大最小より大きくなることはないと思う。
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
    /*
      `img`に順投影を施し、`proj`に得られた値を加える。
      つまり、`proj`は呼び出し元で初期化されている必要がある。
     */
    inner_proj((MatrixXf*)&img, proj, false);
}

void inv_projection(const MatrixXf &proj, MatrixXf *img) {
    /*
      `proj`に逆投影を施し、`img`に得られた値を加える。
      つまり、`img`は呼び出し元で初期化されている必要がある。
     */
    inner_proj(img, (MatrixXf*)&proj, true);
}

void sirt(const MatrixXf &data, MatrixXf *img) {
    /*
      `data`をデータ項としてSIRT法を適用し再構成画像を得る。
      結果は`img`に格納される。
     */
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
        i++;
        if (i % 10 == 0) {
            error = (proj - data).array().abs().sum() / (data.cols() * data.rows());
            printf("%d: %f\n", i, error);
        }
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
