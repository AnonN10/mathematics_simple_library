#include <iostream>
#include <iomanip>
#include <cstdlib>

#include "mathematics_simple_library.hpp"

using namespace Maths;

#define PRINT_EXEC(...) \
std::cout << "> " << #__VA_ARGS__ << std::endl;\
__VA_ARGS__\

template<typename T, std::size_t N>
std::ostream &operator<<(std::ostream &os, const std::array<T, N>& arr) {
	std::string out = "(";
	if(!arr.empty()) {
		for(std::size_t i = 0; i < arr.size() - 1; ++i)
			out += std::to_string(arr[i]) + ", ";
		out += std::to_string(arr.back());
	}
	out += ")";
    return os << out;
}

template<typename T>
std::ostream &operator<<(std::ostream &os, const std::vector<T>& vec) {
	std::string out = "(";
	if(!vec.empty()) {
		for(std::size_t i = 0; i < vec.size() - 1; ++i)
			out += std::to_string(vec[i]) + ", ";
		out += std::to_string(vec.back());
	}
	out += ")";
    return os << out;
}

template <ConceptVector V>
std::ostream &operator<<(std::ostream &os, const V& vec) {
	std::string out = "(";
	if(vec.size().get() > 0) {
		for(std::size_t i = 0; i < vec.size().get() - 1; ++i)
			out += std::to_string(vec[i]) + ", ";
		out += std::to_string(vec[vec.size().get()-1]);
	}
	out += ")";
    return os << out;
}

template <ConceptMatrix M>
std::ostream &operator<<(std::ostream &os, const M& mat) {
	for (IndexType row = 0; row < mat.row_count().get(); ++row) {
		for (IndexType col = 0; col < mat.column_count().get(); ++col)
			os << std::setw(12) << mat[row, col] << ",";
		os << std::endl;
	}
	return os;
}

int main() {
	std::cout << std::fixed;

	std::cout << "## procedural matrix ##" << std::endl;
	PRINT_EXEC(print(mat_identity<4, 5>()));
	PRINT_EXEC(print(mat_zero<2, 2>()));
	PRINT_EXEC(print(mat_one<2, 2>()));
	PRINT_EXEC(print(mat_DFT<4>()));
	PRINT_EXEC(print(mat_DFT(4)));
	PRINT_EXEC(print(mat_walsh_sylvester<int>(8)));
	PRINT_EXEC(print(mat_walsh_sylvester<8, int>()));
	PRINT_EXEC(print(mat_procedural<4, 5>([](auto m, auto n, auto rows, auto columns)->int { return m*columns + n; })));
	std::cout << std::endl;

	std::cout << "## matrix reference (user-defined storage) ##" << std::endl;
	PRINT_EXEC(print(mat_ref<2,3>({1, 2, 3, 4, 5, 6,})));
	PRINT_EXEC(std::array a = {1, 2, 3, 4});
	PRINT_EXEC(std::vector b = {4, 3, 2, 1});
	PRINT_EXEC(print(mat_ref<4,1>(a)));
	PRINT_EXEC(print(mat_ref<1,4>(b)));
	PRINT_EXEC(print(mat_ref<4,1>(a)*mat_ref<1,4>(b)));
	std::cout << std::endl;

	std::cout << "## matrix object (embedded storage: std::array or std::vector) ##" << std::endl;
	PRINT_EXEC(auto mobj_static = mat<2, 2>({1, 2, 3, 4}));
	PRINT_EXEC(auto mobj_dynamic = mat<float>({1, 2, 3, 4}, 2, 2));
	PRINT_EXEC(print(mobj_static));
	PRINT_EXEC(print(mobj_dynamic));
	PRINT_EXEC(mobj_dynamic.resize(1, 2));
	PRINT_EXEC(print(mobj_dynamic));
	PRINT_EXEC(mobj_dynamic.resize(2, 2));
	PRINT_EXEC(print(mobj_dynamic));
	PRINT_EXEC(mobj_dynamic = mat_ref<4,1>(a));
	PRINT_EXEC(print(mobj_dynamic));
	PRINT_EXEC(mobj_dynamic = scaling(vec_ref({1, 2, 3})));
	PRINT_EXEC(print(mobj_dynamic));
	PRINT_EXEC(mobj_dynamic = translation(vec_ref({4, 5, 6})) * scaling(vec_ref({1, 2, 3, 1})));
	PRINT_EXEC(print(mobj_dynamic));
	std::cout << std::endl;
	
	std::cout << "## temporary objects in expressions and underlying type casting via matrix object ##" << std::endl;
	PRINT_EXEC(auto matrix_int = mat<2, 2>({1, 2, 3, 4}));
	PRINT_EXEC(print(matrix_int));
	PRINT_EXEC(print(mat<float>(matrix_int)));
	PRINT_EXEC(matrix_int = mat(matrix_int * matrix_int));
	PRINT_EXEC(print(matrix_int));
	PRINT_EXEC(auto matrix_of_vectors = mat<2, 2, vec_static_t<2, int>>({{-3, -2}, {-1, 0}, {1, 2}, {3, 4}}));
	PRINT_EXEC(print(matrix_of_vectors));
	PRINT_EXEC(auto matrix_of_float_vectors = mat<vec_static_t<2, float>>(matrix_of_vectors));
	PRINT_EXEC(print(matrix_of_float_vectors));
	std::cout << std::endl;

	std::cout << "## unary operators ##" << std::endl;
	PRINT_EXEC(print(-(vec_ref({4, 5, 6}))));
	PRINT_EXEC(print(-translation(vec_ref({4, 5, 6}))));
	PRINT_EXEC(print(unary_operation(vec_ref({1, 2, 3, 4, 5}), [](auto x)->auto { return x*x; })));
	PRINT_EXEC(print(unary_operation(-translation(vec_ref({4, 5, 6})), [](auto x)->auto { return x*x; })));
	std::cout << std::endl;

	std::cout << "## binary operators ##" << std::endl;
	PRINT_EXEC(print(translation(vec_ref({4, 5, 6})) * as_column(vec_ref({1, 2, 3, 1}))));
	PRINT_EXEC(print(translation(vec_ref({4, 5, 6})) * vec_ref({1, 2, 3, 1})));
	PRINT_EXEC(print(as_row(vec_ref({1, 2, 3, 1})) * translation(vec_ref({4, 5, 6}))));
	PRINT_EXEC(print(vec_ref({1, 2, 3, 1}) * transpose(translation(vec_ref({4, 5, 6})))));
	PRINT_EXEC(print(vec_ref({1, 2, 3}) * vec_ref({4, 5, 6})));
	PRINT_EXEC(print(vec_ref({1, 2, 3}) + vec_ref({4, 5, 6})));
	PRINT_EXEC(print(vec_ref({1, 2, 3}) - vec_ref({4, 5, 6})));
	PRINT_EXEC(print(vec_ref({1, 2, 3}) / vec_ref({4, 5, 6})));
	PRINT_EXEC(print(vec_ref({1, 2, 3}) * 2));
	PRINT_EXEC(print(2 * vec_ref({1, 2, 3})));
	PRINT_EXEC(print(vec_ref({1, 2, 3}) / 2));
	PRINT_EXEC(print(vec_ref({1, 2, 3}) + 2));
	PRINT_EXEC(print(vec_ref({1, 2, 3}) - 2));
	PRINT_EXEC(std::cout << dot(vec_ref({1, 2, 3}), vec_ref({4, 5, 6})) << std::endl);
	PRINT_EXEC(print(cross(vec_ref({1, 0, 0}), vec_ref({0, 1, 0}))));
	PRINT_EXEC(print(outer_product(vec_ref({1, 2, 3}), vec_ref({4, 5}))));
	PRINT_EXEC(print(kronecker_product(vec_ref({1, 2, 3}), vec_ref({4, 5}))));
	PRINT_EXEC(print(translation(vec_ref({4, 5, 6})) * 2.0f));
	PRINT_EXEC(print(mat_identity(4,4)/translation(vec_ref({4, 5, 6}))));
	PRINT_EXEC(mat_dynamic_t<float> mat_kp_left = mat({1, 2, 3, 4, 5, 6, 7, 8}, 2, 4));
	PRINT_EXEC(mat_static_t<2,2,float> mat_kp_right = mat<2,2>({10, 20, 30, 40}));
	PRINT_EXEC(print(mat_kp_left));
	PRINT_EXEC(print(mat_kp_right));
	PRINT_EXEC(print(kronecker_product(mat_kp_left, mat_kp_right)));
	PRINT_EXEC(print(hadamard_product(mat({1, 2, 3, 4}, 2, 2), mat({2, 5, 10, 100}, 2, 2))));
	PRINT_EXEC(std::cout << mat(matrix_of_vectors * matrix_of_vectors) << std::endl);
	PRINT_EXEC(std::cout << binary_operation(matrix_of_vectors, vec_static_t<2, int>({1, 2}), [](auto a, auto b)->auto { return dot(a, b); }) << std::endl);
	std::cout << std::endl;

	std::cout << "## ternary operators ##" << std::endl;
	PRINT_EXEC(print(clamp(vec({1, 2, 3}), 0, 2)));
	PRINT_EXEC(std::cout << matrix_of_float_vectors);
	PRINT_EXEC(std::cout << clamp(matrix_of_float_vectors, -1.0f, 1.0f));
	PRINT_EXEC(std::cout << mat<vec_static_t<2, uint8_t>>(clamp(matrix_of_float_vectors, 0.0f, 1.0f) * 255.0f));
	PRINT_EXEC(std::cout << matrix_int);
	PRINT_EXEC(std::cout << ternary_operation(matrix_int, 10, 12345, [](const auto& a, auto b, auto c)->auto { return a > b? a : c; }));
	std::cout << std::endl;
	
	std::cout << "## transformation matrices ##" << std::endl;
	PRINT_EXEC(print(scaling(vec_ref({1, 2, 3}))));
	PRINT_EXEC(print(rotation(vec_ref({1, 0, 0}), vec_ref({0, 1, 0}), 3.14159265358979)));
	PRINT_EXEC(print(translation(vec_ref({4, 5, 6}))));
	PRINT_EXEC(mat_static_t<4,4,float> transform = mat_identity<4, 4>());
	PRINT_EXEC(print(transform));
	PRINT_EXEC(mat_dynamic_t<float> transform_dynamic = mat_identity<4, 4>());
	PRINT_EXEC(print(transform));
	PRINT_EXEC(mat_static_t<4,4,float> m_translation = translation(vec_ref<float>({1, 2, 3})));
	PRINT_EXEC(print(m_translation));
	PRINT_EXEC(mat_static_t<4,4,float> m_rotation = rotation(vec_ref<float>({1, 0, 0, 0}), vec_ref<float>({0, 1, 0, 0}), 3.14159265358979));
	PRINT_EXEC(print(m_rotation));
	PRINT_EXEC(mat_static_t<4,4,float> m_scaling = scaling(vec_ref<float>({1, 2, 3, 1})));
	PRINT_EXEC(print(m_scaling));
	PRINT_EXEC(transform = m_translation * m_rotation * m_scaling);
	PRINT_EXEC(print(transform));
	PRINT_EXEC(transform_dynamic = m_translation * m_rotation * m_scaling);
	PRINT_EXEC(print(transform_dynamic));
	std::cout << std::endl;
	
	std::cout << "## function library ##" << std::endl;
	PRINT_EXEC(std::cout << trace(transform) << std::endl);
	PRINT_EXEC(std::cout << determinant(transform) << std::endl);
	PRINT_EXEC(std::cout << determinant(mat_identity<4, 4>()) << std::endl);
	PRINT_EXEC(std::cout << det(mobj_static) << std::endl);
	PRINT_EXEC(std::cout << det(mobj_dynamic) << std::endl);
	PRINT_EXEC(print(inverse(transform)));
	PRINT_EXEC(print(inv(transform)*transform));
	PRINT_EXEC(print(inverse_gauss_jordan(transform)));
	PRINT_EXEC(print(inverse_gauss_jordan(transform_dynamic)));
	PRINT_EXEC(auto dftmat = mat_DFT<4>());
	PRINT_EXEC(print(transpose(dftmat)));
	PRINT_EXEC(print(transpose_hermitian(dftmat)));
	PRINT_EXEC(print(transpose_hermitian(dftmat)*dftmat));
	PRINT_EXEC(print(augment(mat_multiplicative_identity(4, 4), mat_hadamard_identity(4, 2))));
	PRINT_EXEC(print(split_right(transform, 2)));
	PRINT_EXEC(print(normalize(transform)));
	PRINT_EXEC(print(normalize_max(transform)));
	PRINT_EXEC(print(normalize_minmax(transform)));
	PRINT_EXEC(print(submatrix(transform, 0, 0)));
	PRINT_EXEC(print(adjugate(transform)));
	PRINT_EXEC(print(cofactor(transform)));
	PRINT_EXEC(print(gramian(transform)));
	PRINT_EXEC(print(normalize(vec_ref({1,1,1}))));
	PRINT_EXEC(print(normalize_max(vec_ref<float>({1,5,10}))));
	PRINT_EXEC(print(normalize_minmax(vec_ref<float>({-10,5,10}))));
	PRINT_EXEC(auto incident_ray = vec<float>({-1,1,0}));
	PRINT_EXEC(auto normal = vec<float>({0,1,0}));
	PRINT_EXEC(print(reflect(incident_ray, normal)));
	PRINT_EXEC(print(reflect<Conventions::RayDirection::Outgoing>(incident_ray, normal)));
	PRINT_EXEC(print(reflect<Conventions::RayDirection::Incoming>(-incident_ray, normal)));
	PRINT_EXEC(print(reflect(vec_ref<float>({-1,1,0}), vec_ref<float>({0,1,0}))));
	PRINT_EXEC(decltype(normal)::value_type ior_src = 1.0);
	PRINT_EXEC(decltype(normal)::value_type ior_dest = 1.333);
	PRINT_EXEC(auto eta = ior_src/ior_dest);
	PRINT_EXEC(print(refract(incident_ray, normal, eta)));
	PRINT_EXEC(print(refract<Conventions::RayDirection::Outgoing, true>(incident_ray, normal, eta)));
	PRINT_EXEC(print(refract<Conventions::RayDirection::Incoming, true>(-incident_ray, normal, eta)));
	
    return EXIT_SUCCESS;
}
