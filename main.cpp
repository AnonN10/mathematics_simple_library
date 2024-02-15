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
	std::cout << "a.euclidean_normalize():" << std::endl;
	print_matrix(a.euclidean_normalize());
	std::cout << "c.euclidean_normalize():" << std::endl;
	print_matrix(c.euclidean_normalize());
	
	std::cout << "m:" << std::endl;
	print_matrix(m);
	std::cout << "m.euclidean_normalize():" << std::endl;
	print_matrix(m.euclidean_normalize());
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
	
	
    return 0;
}