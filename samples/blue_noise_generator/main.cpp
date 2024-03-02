#include <iostream>
#include <iomanip>

#include <cstdlib>
#include <random>

#include "mathematics_simple_library.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace Maths;

template<typename T>
T random_range(T range_from, T range_to) {
    std::random_device                  rand_dev;
    std::mt19937                        generator(rand_dev());
    std::uniform_real_distribution<T>   distr(range_from, range_to);
    return distr(generator);
}

int main() {
	constexpr int matdim = 512;
	using complex_value_type = double;
	Matrix<matdim,matdim,std::complex<complex_value_type>,true> mat;
	Matrix<matdim,matdim,std::complex<complex_value_type>,true> mat_filter;
	Matrix<matdim,matdim,std::complex<complex_value_type>,true> mat_DFT;
	Matrix<matdim,matdim,std::complex<complex_value_type>,true> mat_iDFT;
	std::cout << "Generating DFT and iDFT matrices..." << std::endl;
    mat_DFT.DFT();
    mat_iDFT = mat_DFT.transpose_hermitian() / complex_value_type(matdim);
	std::cout << "Generating white noise matrix..." << std::endl;
	for(IndexType m = 0; m < matdim; ++m)
		for(IndexType n = 0; n < matdim; ++n) {
			mat[m, n] = random_range(complex_value_type(0), complex_value_type(1));
		}
	std::cout << "Generating filter matrix..." << std::endl;
	auto sqr = [](auto x) { return x*x; };
	for(IndexType m = 0; m < matdim; ++m)
		for(IndexType n = 0; n < matdim; ++n) {
			mat_filter[m, n] = complex_value_type(1);
			complex_value_type radius = matdim/4;
			mat_filter[m, n] *= std::sqrt(
				static_cast<float>(sqr(static_cast<complex_value_type>(n)) + sqr(static_cast<complex_value_type>(m)))
			) > radius? 1.0 : 0.0;
			mat_filter[m, n] *= std::sqrt(
				static_cast<float>(sqr(static_cast<complex_value_type>(matdim - n)) + sqr(static_cast<complex_value_type>(m)))
			) > radius? 1.0 : 0.0;
			mat_filter[m, n] *= std::sqrt(
				static_cast<float>(sqr(static_cast<complex_value_type>(n)) + sqr(static_cast<complex_value_type>(matdim - m)))
			) > radius? 1.0 : 0.0;
			mat_filter[m, n] *= std::sqrt(
				static_cast<float>(sqr(static_cast<complex_value_type>(matdim - n)) + sqr(static_cast<complex_value_type>(matdim - m)))
			) > radius? 1.0 : 0.0;
		}
		
	std::cout << "Transforming to frequency domain..." << std::endl;
	mat = (mat_DFT * mat) * mat_DFT.transpose();
	std::cout << "Applying filter..." << std::endl;
	mat = mat.hadamard_product(mat_filter);
	std::cout << "Transforming to time domain..." << std::endl;
	mat = (mat_iDFT * mat) * mat_iDFT.transpose();
    
    std::vector<uint8_t> image_data(matdim * matdim);
	
	std::cout << "Reading out the data..." << std::endl;
    Matrix<matdim,matdim,complex_value_type,true> mat_out;
	for(IndexType m = 0; m < matdim; ++m)
		for(IndexType n = 0; n < matdim; ++n) {
			mat_out[m, n] = mat[m, n].real();
		}
	std::cout << "Transforming to image space..." << std::endl;
	mat_out = mat_out.normalize_minmax();
    auto matrix_elements = mat_out.get_v();
    std::transform(
		matrix_elements.begin(),
		matrix_elements.end(),
		image_data.begin(),
		[](auto in)->uint8_t{
			return static_cast<uint8_t>(std::min(std::max(in, complex_value_type(0)), complex_value_type(1))*255.0);
		}
	);
	
	std::cout << "Saving to bluenoise.png..." << std::endl;
	stbi_write_png("bluenoise.png", matdim, matdim, 1, image_data.data(), 0);

	std::cout << "Done." << std::endl;
	
    return 0;
}