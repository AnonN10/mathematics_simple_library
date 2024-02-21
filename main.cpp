#include <iostream>
#include <iomanip>

#include "mathematics_simple_library.hpp"

using namespace Maths;

template<IndexType M, IndexType N, typename Scalar>
void print_matrix(const Matrix<M, N, Scalar>& mat) {
	for(IndexType m = 0; m < M; ++m) {
		for(IndexType n = 0; n < N; ++n)
			std::cout << std::setw(12) << mat.data[m][n] << ",";
		std::cout << std::endl;
	}
}

int main() {
	std::cout << std::fixed;
	
	Matrix<2,3> m({
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
	
    return 0;
}
