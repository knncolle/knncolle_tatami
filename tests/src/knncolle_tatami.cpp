#include <gtest/gtest.h>

#include "knncolle_tatami/knncolle_tatami.hpp"

#include <numeric>
#include <random>
#include <vector>
#include <algorithm>
#include <cstddef>

TEST(Matrix, Simple) {
    int NR = 20, NC = 50;
    std::vector<double> buffer(NR * NC);
    std::iota(buffer.begin(), buffer.end(), 0);

    auto dmat = std::make_shared<tatami::DenseMatrix<double, int, std::vector<double> > >(NR, NC, buffer, false);
    knncolle_tatami::Matrix<int, double, double, int> tmat(std::move(dmat), false);
    EXPECT_EQ(tmat.num_dimensions(), NR);
    EXPECT_EQ(tmat.num_observations(), NC);

    std::vector<double> tmp(NR), tmp2(NR);
    auto work = tmat.new_extractor();
    for (int c = 0; c < NC; ++c) {
        auto it = buffer.begin() + static_cast<std::size_t>(c) * static_cast<std::size_t>(NR);
        std::copy_n(it, NR, tmp.begin());
        auto ptr = work->next();
        std::copy_n(ptr, NR, tmp2.begin());
        EXPECT_EQ(tmp, tmp2);
    }
}

TEST(Matrix, DifferentType) {
    int NR = 20, NC = 50;
    std::vector<float> buffer(NR * NC);
    std::iota(buffer.begin(), buffer.end(), 0);

    auto dmat = std::make_shared<tatami::DenseMatrix<double, int, std::vector<float> > >(NR, NC, buffer, false);
    knncolle_tatami::Matrix<int, double, double, int> tmat(std::move(dmat), false);
    EXPECT_EQ(tmat.num_dimensions(), NR);
    EXPECT_EQ(tmat.num_observations(), NC);

    std::vector<double> tmp(NR), tmp2(NR);
    auto work = tmat.new_extractor();
    for (int c = 0; c < NC; ++c) {
        auto it = buffer.begin() + static_cast<std::size_t>(c) * static_cast<std::size_t>(NR);
        std::copy_n(it, NR, tmp.begin());
        auto ptr = work->next();
        std::copy_n(ptr, NR, tmp2.begin());
        EXPECT_EQ(tmp, tmp2);
    }
}

TEST(Matrix, Transposed) {
    int NR = 20, NC = 50;
    std::vector<double> buffer(NR * NC);
    std::iota(buffer.begin(), buffer.end(), 0);

    auto dmat = std::make_shared<tatami::DenseMatrix<double, int, std::vector<double> > >(NR, NC, buffer, true);
    knncolle_tatami::Matrix<int, double, double, int> tmat(std::move(dmat), true);
    EXPECT_EQ(tmat.num_dimensions(), NC);
    EXPECT_EQ(tmat.num_observations(), NR);

    std::vector<double> tmp(NC), tmp2(NC);
    auto work = tmat.new_extractor();
    for (int r = 0; r < NR; ++r) {
        auto it = buffer.begin() + static_cast<std::size_t>(r) * static_cast<std::size_t>(NC);
        std::copy_n(it, NC, tmp.begin());
        auto ptr = work->next();
        std::copy_n(ptr, NC, tmp2.begin());
        EXPECT_EQ(tmp, tmp2);
    }
}

TEST(Neighbors, Full) {
    int NR = 20, NC = 500;
    std::vector<double> buffer(NR * NC);
    std::mt19937_64 rng(1234567);
    std::normal_distribution ndist;
    for (auto& b : buffer) {
        b = ndist(rng);
    }

    auto dmat = std::make_shared<tatami::DenseMatrix<double, int, std::vector<double> > >(NR, NC, buffer, false);
    knncolle_tatami::Matrix<int, double, double, int> tmat(std::move(dmat), false);
    knncolle::SimpleMatrix<int, double> smat(NR, NC, buffer.data());
    const int k = 10;

    knncolle::VptreeBuilder<int, double, double> builder(std::make_shared<knncolle::EuclideanDistance<double, double> >());
    auto tdex = builder.build_unique(tmat);
    auto sdex = builder.build_unique(smat);

    auto tres = find_nearest_neighbors(*tdex, k);
    auto sres = find_nearest_neighbors(*sdex, k);

    for (int c = 0; c < NC; ++c) {
        EXPECT_EQ(tres[c], sres[c]);
    }
}
