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
	constexpr auto zero = complex_value_type{0};
	constexpr auto one = complex_value_type{1};
	std::cout << "Generating DFT and iDFT matrices..." << std::endl;
    mat_dynamic_t<std::complex<complex_value_type>> mat_DFT = Maths::mat_DFT<matdim, complex_value_type>();
    mat_dynamic_t<std::complex<complex_value_type>> mat_iDFT = transpose_hermitian(mat_DFT);
	std::cout << "Generating white noise matrix..." << std::endl;
	mat_dynamic_t<std::complex<complex_value_type>> mat = mat_procedural(
		matdim, matdim,
		[](auto, auto, auto, auto)->auto { return random_range(zero, one); }
	);
	std::cout << "Generating filter matrix..." << std::endl;
	mat_dynamic_t<std::complex<complex_value_type>> mat_filter = mat_procedural(
		matdim, matdim,
		[](auto m, auto n, auto rows, auto columns) {
			complex_value_type center_offset = matdim/2;
			complex_value_type radius = matdim/4;
			auto x = fftshift(static_cast<complex_value_type>(n), static_cast<complex_value_type>(matdim));
			auto y = fftshift(static_cast<complex_value_type>(m), static_cast<complex_value_type>(matdim));
			return norm_frobenius(vec_ref<complex_value_type>({x, y})-center_offset) > radius? 1.0 : 0.0;
		}
	);
		
	// since expression templates operate on the data in-place,
	// it's important to take caution for cases where you assign
	// to the matrix object while operating on its contents,
	// therefore a temporary object is constructed using Maths::mat()
	// to prevent reading and writing to the same memory simultaneously
	std::cout << "Transforming to frequency domain..." << std::endl;
	mat = Maths::mat(mat_DFT * mat) * transpose(mat_DFT);
	std::cout << "Applying filter..." << std::endl;
	mat = Maths::mat(hadamard_product(mat, mat_filter));
	std::cout << "Transforming to time domain..." << std::endl;
	mat = Maths::mat(mat_iDFT * mat) * transpose(mat_iDFT);
	
	std::cout << "Reading out the data..." << std::endl;
    mat_dynamic_t<complex_value_type> mat_real = unary_operation(mat, [](const auto& x)->auto { return x.real(); });
	std::cout << "Transforming to image space..." << std::endl;
	auto mat_image_data = Maths::mat<uint8_t>(clamp(normalize_minmax(mat_real), zero, one) * 255);
	
	std::cout << "Saving to bluenoise.png..." << std::endl;
	stbi_write_png("bluenoise.png", matdim, matdim, 1, mat_image_data.data.data(), 0);

	std::cout << "Done." << std::endl;
	
    return 0;
}