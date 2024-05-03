#include <iostream>
#include <iomanip>

#include <cstdlib>
#include <random>
#include <tuple>
#include <vector>
#include <type_traits>
#include <memory>

#include "mathematics_simple_library.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace Maths;

template<typename T>
T random_range(T range_from, T range_to) {
    std::random_device                     rand_dev;
    std::mt19937                           generator(rand_dev());
    std::conditional_t<
		std::is_integral_v<T>,
		std::uniform_int_distribution<T>,
		std::uniform_real_distribution<T>> distr(range_from, range_to);
    return distr(generator);
}

template<typename T>
constexpr auto distance_metric(T x) {
	return norm_euclidean(x);
	//return norm_chebyshev(x);
	//return norm_manhattan(x);
}

int main() {
	constexpr std::size_t
		image_width = 1024,
		image_height = 1024,
		seed_count = 64;

	using seed_type = std::tuple<
		vec_static_t<2, std::int32_t>,//position (x, y), in image space [0; width)*[0; height)
		vec_static_t<3, float>//color
	>;
	std::array<seed_type, seed_count> seeds;
	
	using cell_type = std::tuple<
		seed_type*,//assigned seed
		float//distance to assigned seed, in normalized resolution space [0; width)*[0; height)/max(width, height)
	>;
	//a matrix to store the distance field and the closest seed at each cell
	mat_dynamic_t<cell_type> mat_image(image_height, image_width);

	//insert seeds into the image matrix at random positions
	for(auto&& seed : seeds) {
		auto position = vec({
				random_range<std::int32_t>(0, image_width-1),
				random_range<std::int32_t>(0, image_height-1),
		});
		seed = seed_type(
			position,
			vec_procedural<3>([](auto, auto)->auto { return random_range<float>(0.0, 1.0); } )
		);
		//row index is y, column index is x - hence y, x
		auto&& cell = mat_image[position[1], position[0]];
		//insert the seed into the cell
		std::get<0>(cell) = &seed;
		std::get<1>(cell) = 0.0;
	}

	const std::size_t max_dimension = std::max(image_width, image_height);
	const std::size_t iterations = std::log2(max_dimension);

	std::int32_t offset_scale = std::max(image_width, image_height);
	for(IndexType i = 0; i < iterations; ++i) {
		offset_scale /= 2;
		std::cout << "offset: " << offset_scale << std::endl;
		//for each element of the image matrix
		for(IndexType m = 0; m < image_height; ++m)
			for(IndexType n = 0; n < image_width; ++n) {
				auto&& current_cell = mat_image[m, n];
				vec_static_t<2, float> cell_pos = vec<float>(vec_ref({n, m}));
				//for each 3x3 kernel sample
				for(std::int32_t y = -1; y <= 1; ++y) {
					for(std::int32_t x = -1; x <= 1; ++x) {
						//ignore self
						if(x==0 && y==0) continue;
						//sample cell at iteration's offset y, x
						auto&& sample_cell = mat_image[
							circshift<std::int32_t>(m, x*offset_scale, image_width),
							circshift<std::int32_t>(n, y*offset_scale, image_height)
						];
						//if sample is not empty
						auto sample_seed = std::get<0>(sample_cell);
						if(sample_seed) {
							//compute distance from current cell position to the sample's seed position in normalized resolution space
							auto seed_pos = vec<float>(std::get<0>(*sample_seed));
							float distance = distance_metric((seed_pos-cell_pos)/static_cast<float>(max_dimension));
							auto min_distance = std::get<1>(current_cell);
							//if distance is smaller or is currently empty
							if(distance < min_distance || !std::get<0>(current_cell)) {
								//assign seed and the distance to it
								std::get<0>(current_cell) = sample_seed;
								std::get<1>(current_cell) = distance;
							}
						}
					}
				}
			}
	}
	
	std::cout << "Transforming to image space..." << std::endl;
	auto mat_image_data_voronoi = mat<vec_static_t<3, std::uint8_t>>(mat(unary_operation(
		mat_image,
		[](const auto& in)->auto {
			if(!std::get<0>(in)) return std::remove_reference_t<decltype(std::get<1>(*std::get<0>(in)))>{};
			return std::get<1>(*std::get<0>(in));
		}
	))*255);
	auto mat_image_data_distance_field = mat<std::uint8_t>((unary_operation(
		mat_image,
		[](const auto& in)->auto { return std::get<1>(in); }
	))*255);
    
	std::cout << "Saving to voronoi.png..." << std::endl;
	stbi_write_png("voronoi.png", image_width, image_height, 3, mat_image_data_voronoi.data.data(), 0);
    
	std::cout << "Saving to distancefield.png..." << std::endl;
	stbi_write_png("distancefield.png", image_width, image_height, 1, mat_image_data_distance_field.data.data(), 0);

	std::cout << "Done." << std::endl;
	
    return EXIT_SUCCESS;
}