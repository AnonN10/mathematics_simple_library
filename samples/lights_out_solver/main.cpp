#include <iostream>
#include <iomanip>

#include "mathematics_simple_library.hpp"

using namespace Maths;

struct GF2 {
	bool value = 0;
	GF2() = default;
	GF2(bool v) { value = !!v; }
	GF2 operator+(const GF2& other) const {
		return GF2(value ^ other.value);
	}
	GF2 operator-(const GF2& other) const {
		return (*this)+other;
	}
	GF2 operator*(const GF2& other) const {
		return GF2(value * other.value);
	}
	GF2 operator/(const GF2& other) const {
		return GF2(value / other.value);
	}
	void operator+=(const GF2& other) {
		value = ((*this)+other).value;
	}
	void operator-=(const GF2& other) {
		value = ((*this)-other).value;
	}
	void operator*=(const GF2& other) {
		value = ((*this)*other).value;
	}
	void operator/=(const GF2& other) {
		value = ((*this)/other).value;
	}
	bool operator<(const GF2& other) const {
		return value < other.value;
	}
	bool operator>(const GF2& other) const {
		return value > other.value;
	}
	bool operator==(const GF2& other) const {
		return value == other.value;
	}
	bool operator!=(const GF2& other) const {
		return value != other.value;
	}
};
GF2 abs(GF2 x) { return x; }
std::ostream &operator<<(std::ostream &os, GF2 const &m) { 
    return os << m.value;
}

template<IndexType M, IndexType N, typename Scalar, bool use_heap>
void print_matrix(const Matrix<M, N, Scalar, use_heap>& mat, std::streamsize spacing_width = 12) {
	for(IndexType m = 0; m < M; ++m) {
		for(IndexType n = 0; n < N; ++n)
			std::cout << std::setw(spacing_width) << mat[m, n] << ",";
		std::cout << std::endl;
	}
}

int main() {
	std::cout << std::fixed;

	Matrix<5,5,GF2> board({
		GF2(0), GF2(1), GF2(0), GF2(1), GF2(1),
		GF2(1), GF2(1), GF2(0), GF2(0), GF2(0),
		GF2(0), GF2(1), GF2(1), GF2(1), GF2(1),
		GF2(0), GF2(0), GF2(1), GF2(0), GF2(1),
		GF2(0), GF2(1), GF2(0), GF2(1), GF2(0),
	});

	std::vector<GF2> A_flat;
	for(int m = 0; m < 5; ++m) {
		for(int n = 0; n < 5; ++n) {
			Matrix<5,5,GF2> btnpress;
			btnpress.zero();
			btnpress[m, n] = GF2(1);
			if(m > 0) btnpress[m-1, n] = GF2(1);
			if(m < 5-1) btnpress[m+1, n] = GF2(1);
			if(n > 0) btnpress[m, n-1] = GF2(1);
			if(n < 5-1) btnpress[m, n+1] = GF2(1);
			auto btnpress_flat = btnpress.get_v();
			A_flat.insert(A_flat.end(), btnpress_flat.begin(), btnpress_flat.end());
		}
	}
	//construct row matrix
	Matrix<5*5,5*5,GF2,true> A_T;
	A_T.set(A_flat);
	//transpose such that each board state corresponds to a column in matrix A
	auto A = A_T.transpose();
	//turn initial board state to column vector
	Vector<5*5,GF2,true> b;
	b.set(board.get_v());
	//augment
	auto M = A.augment(b);
	//solve
	M = M.rref();
	//extract solution and turn back to 5x5 matrix
	Matrix<5,5,GF2> solution;
	solution.set(M.column(25).get_v());

	//print initial board state and its solution
	std::cout << "board:" << std::endl;
	print_matrix(board, 2);
	std::cout << "solution:" << std::endl;
	print_matrix(solution, 2);
	
    return 0;
}
