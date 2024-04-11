#include <iostream>
#include <iomanip>
#include <cstdlib>

#include "mathematics_simple_library.hpp"

using namespace Maths;

#define PRINT_EXEC(...) \
std::cout << "> " << #__VA_ARGS__ << std::endl;\
__VA_ARGS__\

int main() {
	std::cout << std::fixed;

	std::cout << "## procedural matrix ##" << std::endl;
	PRINT_EXEC(print(mat_identity<4, 5>()));
	PRINT_EXEC(print(mat_zero<2, 2>()));
	PRINT_EXEC(print(mat_one<2, 2>()));
	PRINT_EXEC(print(mat_DFT<4>()));
	PRINT_EXEC(print(mat_DFT(4)));
	PRINT_EXEC(print(mat_walsh_sylvester<int>(8)));
	std::cout << std::endl;

	std::cout << "## matrix reference (user-defined storage) ##" << std::endl;
	PRINT_EXEC(print(mat_ref<2,3>({1, 2, 3, 4, 5, 6,})));

	PRINT_EXEC(std::array a = {1, 2, 3, 4});
	PRINT_EXEC(std::array b = {4, 3, 2, 1});
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

	std::cout << "# binary operators #" << std::endl;
	PRINT_EXEC(print(translation(vec_ref({4, 5, 6})) * as_column(vec_ref({1, 2, 3, 1}))));
	PRINT_EXEC(print(translation(vec_ref({4, 5, 6})) * vec_ref({1, 2, 3, 1})));
	PRINT_EXEC(print(as_row(vec_ref({1, 2, 3, 1})) * translation(vec_ref({4, 5, 6}))));
	PRINT_EXEC(print(vec_ref({1, 2, 3, 1}) * transpose(translation(vec_ref({4, 5, 6})))));
	PRINT_EXEC(std::cout << dot(vec_ref({1, 2, 3}), vec_ref({4, 5, 6})) << std::endl);
	PRINT_EXEC(print(translation(vec_ref({4, 5, 6})) * 2.0f));
	PRINT_EXEC(print(mat_identity(4,4)/translation(vec_ref({4, 5, 6}))));

	PRINT_EXEC(mat_dynamic_t<float> mat_kp_left = mat({1, 2, 3, 4, 5, 6, 7, 8}, 2, 4));
	PRINT_EXEC(mat_dynamic_t<float> mat_kp_right = mat({10, 20, 30, 40}, 2, 2));
	PRINT_EXEC(print(mat_kp_left));
	PRINT_EXEC(print(mat_kp_right));
	PRINT_EXEC(print(kronecker_product(mat_kp_left, mat_kp_right)));
	
	PRINT_EXEC(print(hadamard_product(mat({1, 2, 3, 4}, 2, 2), mat({2, 5, 10, 100}, 2, 2))));
	std::cout << std::endl;
	
	std::cout << "# transformation matrices #" << std::endl;
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
	PRINT_EXEC(mat_static_t<4,4,float> m_scaling = scaling(vec_ref<float>({2, 3, 4, 1})));
	PRINT_EXEC(print(m_scaling));
	PRINT_EXEC(transform = m_translation * m_rotation * m_scaling);
	PRINT_EXEC(print(transform));
	PRINT_EXEC(transform_dynamic = m_translation * m_rotation * m_scaling);
	PRINT_EXEC(print(transform_dynamic));
	std::cout << std::endl;
	
	std::cout << "# function library #" << std::endl;

	PRINT_EXEC(std::cout << determinant(transform) << std::endl);
	PRINT_EXEC(std::cout << determinant(mat_identity<4, 4>()) << std::endl);
	PRINT_EXEC(std::cout << det(mobj_static) << std::endl);
	PRINT_EXEC(std::cout << det(mobj_dynamic) << std::endl);

	PRINT_EXEC(print(inverse(transform)));
	PRINT_EXEC(print(inv(transform)*transform));
	PRINT_EXEC(print(inverse_gauss_jordan(transform)));
	PRINT_EXEC(print(inverse_gauss_jordan(transform_dynamic)));

	/*mat_dynamic_t<float> mattest({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}, 4, 4);
	print(mattest);
	print(submatrix(mattest, 2, 0));
	print(submatrix(submatrix(mattest, 2, 0), 0, 0));*/
	
	/*Matrix<2,3> m({
		1, 2, 3,
		4, 5, 6,
	});
	Vector<4> a({1, 2, 3, 4});
	Vector<4> b({4, 3, 2, 1});
	Covector<4> c = a.transpose();
	Covector<4> d = b.transpose();
	
	std::cout << "a:" << std::endl;
	print_matrix(a);
	std::cout << "b:" << std::endl;
	print_matrix(b);
	std::cout << "c:" << std::endl;
	print_matrix(c);
	std::cout << "d:" << std::endl;
	print_matrix(d);
	
	std::cout << "dot(a, b)=" << a.dot(b) << std::endl;
	std::cout << "dot(c, d)=" << c.dot(d) << std::endl;
	std::cout << "a*c:"<< std::endl;
	print_matrix(a*c);
	std::cout << "b*d:"<< std::endl;
	print_matrix(b*d);
	std::cout << "a.normalize_euclidean():" << std::endl;
	print_matrix(a.normalize_euclidean());
	std::cout << "c.normalize_euclidean():" << std::endl;
	print_matrix(c.normalize_euclidean());
	
	std::cout << "m:" << std::endl;
	print_matrix(m);
	std::cout << "m.normalize_euclidean():" << std::endl;
	print_matrix(m.normalize_euclidean());
	std::cout << "m.gramian():" << std::endl;
	print_matrix(m.gramian());
	
	Matrix<2,2> A({
		1, 2,
		3, 4,
	});
	Matrix<2,2> B({
		0, 5,
		6, 7,
	});
	std::cout << "A:" << std::endl;
	print_matrix(A);
	std::cout << "B:" << std::endl;
	print_matrix(B);
	std::cout << "A.kronecker_product(B):" << std::endl;
	print_matrix(A.kronecker_product(B));
	std::cout << "A.augment(B):" << std::endl;
	print_matrix(A.augment(B));
	std::cout << "A.augment(B).split_right<2>():" << std::endl;
	print_matrix(A.augment(B).split_right<2>());
	
	Matrix<4,4,std::complex<float>> DFT, iDFT;
	DFT.DFT();
	iDFT = DFT.inverse();
	std::cout << "DFT:" << std::endl;
	print_matrix(DFT);
	std::cout << "iDFT:" << std::endl;
	print_matrix(iDFT);
	Vector<4, std::complex<float>> a_complex = static_cast<Vector<4, std::complex<float>>>(a);
	std::cout << "DFT*a:" << std::endl;
	print_matrix(DFT*a_complex);
	std::cout << "iDFT*(DFT*a):" << std::endl;
	print_matrix(iDFT*(DFT*a_complex));
	
	std::cout << "Walsh n=2:" << std::endl;
	Matrix<2,2> walsh2;
	walsh2.sylvester_walsh();
	print_matrix(walsh2);
	std::cout << "Walsh n=4:" << std::endl;
	Matrix<4,4> walsh4;
	walsh4.sylvester_walsh();
	print_matrix(walsh4);
	std::cout << "Walsh n=8:" << std::endl;
	Matrix<8,8> walsh8;
	walsh8.sylvester_walsh();
	print_matrix(walsh8);

	Matrix<3,4,double> abc({
		5, -6, -7,   7,
        3, -2,  5, -17,
        2,  4, -3,  29
	});
	std::cout << "abc:" << std::endl;
	print_matrix(abc);
	std::cout << "abc.rref():" << std::endl;
	print_matrix(abc.rref());
	std::cout << "abc.split_right<1>():" << std::endl;
	print_matrix(abc.split_right<1>());
	std::cout << "abc.split_right<1>().inverse_gauss_jordan():" << std::endl;
	print_matrix(abc.split_right<1>().inverse_gauss_jordan());
	std::cout << "abc.split_right<1>().inverse_gauss_jordan()*abc.split_right<1>():" << std::endl;
	print_matrix(abc.split_right<1>().inverse_gauss_jordan()*abc.split_right<1>());

	std::cout << "abc.normalize_minmax():" << std::endl;
	print_matrix(abc.normalize_minmax());

	std::transform(abc.begin(), abc.end(), abc.begin(), [](auto in)->auto { return in*2.0; });
	std::cout << "abc*2 via std::transform(...):" << std::endl;
	print_matrix(abc);*/
	
    return EXIT_SUCCESS;
}
