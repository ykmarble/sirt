#include <cstdio>
#include <cstdlib>
#include <memory>
#include <eigen3/Eigen/Core>

namespace {

void load_rawimage(const char *path, Eigen::MatrixXf *img) {
    FILE* f = fopen(path, "rb");
    char magic[4];
    unsigned int width, height;
    fread(magic, sizeof(char), 4, f);
    fread(&width, sizeof(unsigned int), 1, f);
    fread(&height, sizeof(unsigned int), 1, f);
    std::unique_ptr<float> img_seq(new float[width * height]);
    fread(img_seq.get(), sizeof(float), width * height, f);
    fclose(f);
    *img = Eigen::Map<Eigen::MatrixXf> (img_seq.get(), width, height);
}

void save_rawimage(const char *path, Eigen::MatrixXf &img) {
    char magic[] = {'P', '0', 0x00, 0x00};
    unsigned int width = img.cols();
    unsigned int height = img.rows();
    std::unique_ptr<float> img_seq(new float[width * height]);
    Eigen::Map<Eigen::MatrixXf>(img_seq.get(), width, height) = img;
    FILE* f = fopen(path, "wb");
    fwrite(magic, sizeof(char), 4, f);
    fwrite(&width, sizeof(unsigned int), 1, f);
    fwrite(&height, sizeof(unsigned int), 1, f);
    fwrite(img_seq.get(), sizeof(float), width * height, f);
    fclose(f);
}
}

int main(int argc, char** argv) {
    if (argc != 2) {
        printf("Usage %s file", argv[0]);
        return 1;
    }
    Eigen::MatrixXf img;
    load_rawimage(argv[1], &img);
    save_rawimage("out.dat", img);
    return 0;
}
