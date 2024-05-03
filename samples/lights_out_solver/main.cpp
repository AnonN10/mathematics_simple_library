#include <iostream>
#include <iomanip>
#include <cstdlib>

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

int main() {
	std::cout << std::fixed;

	auto board = mat<5,5,GF2>({
		GF2(0), GF2(1), GF2(0), GF2(1), GF2(1),
		GF2(1), GF2(1), GF2(0), GF2(0), GF2(0),
		GF2(0), GF2(1), GF2(1), GF2(1), GF2(1),
		GF2(0), GF2(0), GF2(1), GF2(0), GF2(1),
		GF2(0), GF2(1), GF2(0), GF2(1), GF2(0),
	});

	std::vector<GF2> A_flat;
	for(int m = 0; m < 5; ++m) {
		for(int n = 0; n < 5; ++n) {
			mat_static_t<5,5,GF2> btnpress = mat_zero<5,5,GF2>();
			btnpress[m, n] = GF2(1);
			if(m > 0) btnpress[m-1, n] = GF2(1);
			if(m < 5-1) btnpress[m+1, n] = GF2(1);
			if(n > 0) btnpress[m, n-1] = GF2(1);
			if(n < 5-1) btnpress[m, n+1] = GF2(1);
			A_flat.insert(A_flat.end(), btnpress.data.begin(), btnpress.data.end());
		}
	}
	//construct row matrix and transpose such that each board state corresponds to a column in matrix A
	auto A = transpose(mat_ref(A_flat, 5*5, 5*5));
	//turn initial board state to column vector
	auto b = as_column(vec_ref(board.data));
	//augment
	mat_dynamic_t<GF2> M = augment(A, b);
	//solve
	M = rref(M);
	//extract solution and turn back to 5x5 matrix
	auto solution = as_matrix(column_of(M, 25), 5);

	//print initial board state and its solution
	std::cout << "board:" << std::endl;
	print(board, std::cout, 2);
	std::cout << "solution:" << std::endl;
	print(solution, std::cout, 2);
	
    return EXIT_SUCCESS;
}
