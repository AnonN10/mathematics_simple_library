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

float distance_metric(float x, float y) {
	//Euclidean distance
	return std::sqrt(x*x+y*y);
	//Chebyshev distance
	//return std::max(std::abs(x), std::abs(y));
	//Manhattan distance
	//return std::abs(x)+std::abs(y);
}

int main() {
	constexpr std::size_t
		image_width = 1024,
		image_height = 1024,
		seed_count = 64;
	using seed_type = std::tuple<
		std::array<std::int32_t, 2>,//position (x, y), in image space [0; width)*[0; height)
		std::array<float, 3>//color
	>;
	using cell_type = std::tuple<
		seed_type*,//assigned seed
		float//distance to assigned seed, in normalized resolution space [0; width)*[0; height)/max(width, height)
	>;
	//a matrix to store the distance field and the closest seed at each cell
	mat_dynamic_t<cell_type>/*Matrix<image_height, image_width, cell_type, true>*/ mat_image;
	mat_image.resize(image_height, image_width);
	std::array<seed_type, seed_count> seeds;
	for(auto&& seed : seeds) {
		auto position = std::array<std::int32_t, 2>({
			random_range<std::int32_t>(0, image_width-1),
			random_range<std::int32_t>(0, image_height-1),
		});
		seed = seed_type(
			position,
			{
				random_range<float>(0.0, 1.0),
				random_range<float>(0.0, 1.0),
				random_range<float>(0.0, 1.0),
			}
		);
		//row index is y, column index is x - hence y, x
		auto&& cell = mat_image[position[1], position[0]];
		//insert seed into the cell
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
				//for each 3x3 kernel sample
				for(std::int32_t y = -1; y <= 1; ++y) {
					for(std::int32_t x = -1; x <= 1; ++x) {
						//ignore self
						if(x==0 && y==0) continue;
						//sample cell at iteration's offset y, x
						IndexType
							pos_x = eucmod(static_cast<std::int32_t>(n)+x*offset_scale, static_cast<std::int32_t>(image_width)),
							pos_y = eucmod(static_cast<std::int32_t>(m)+y*offset_scale, static_cast<std::int32_t>(image_height));
						auto&& sample_cell = mat_image[pos_y, pos_x];
						//if sample is not empty
						auto sample_seed = std::get<0>(sample_cell);
						if(sample_seed) {
							//compute distance from current cell position to the sample's seed position in normalized resolution space
							auto seed_pos = std::get<0>(*sample_seed);
							float distance = distance_metric(
								static_cast<float>(seed_pos[0]-static_cast<std::int32_t>(n))/static_cast<float>(max_dimension),
								static_cast<float>(seed_pos[1]-static_cast<std::int32_t>(m))/static_cast<float>(max_dimension)
							);
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
    
    std::vector<std::tuple<std::uint8_t, std::uint8_t, std::uint8_t>> image_data_voronoi(image_width * image_height);
    std::vector<std::uint8_t> image_data_distance_field(image_width * image_height);
	
	std::cout << "Transforming to image space..." << std::endl;
    std::transform(
		mat_image.data.begin(),
		mat_image.data.end(),
		image_data_voronoi.begin(),
		[](auto in)->std::tuple<std::uint8_t, std::uint8_t, std::uint8_t> {
			std::tuple<std::uint8_t, std::uint8_t, std::uint8_t> ret;
			if(!std::get<0>(in)) return ret;
			auto&& seed = *std::get<0>(in);
			auto clamp_normalize = [](auto x)->std::uint8_t {
				return static_cast<std::uint8_t>(
					std::min(
						std::max(x, static_cast<decltype(x)>(0)),
						static_cast<decltype(x)>(1)
					)*static_cast<decltype(x)>(255)
				);
			};
			std::get<0>(ret) = clamp_normalize(std::get<1>(seed)[0]);
			std::get<1>(ret) = clamp_normalize(std::get<1>(seed)[1]);
			std::get<2>(ret) = clamp_normalize(std::get<1>(seed)[2]);
			return ret;
		}
	);
	std::transform(
		mat_image.data.begin(),
		mat_image.data.end(),
		image_data_distance_field.begin(),
		[](auto in)->std::uint8_t {
			auto dist = std::get<1>(in);
			return static_cast<std::uint8_t>(
				std::min(
					std::max(dist, static_cast<decltype(dist)>(0)),
					static_cast<decltype(dist)>(1)
				)*static_cast<decltype(dist)>(255));
		}
	);
    
	std::cout << "Saving to voronoi.png..." << std::endl;
	stbi_write_png("voronoi.png", image_width, image_height, 3, image_data_voronoi.data(), 0);
    
	std::cout << "Saving to distancefield.png..." << std::endl;
	stbi_write_png("distancefield.png", image_width, image_height, 1, image_data_distance_field.data(), 0);

	std::cout << "Done." << std::endl;
	
    return 0;
}