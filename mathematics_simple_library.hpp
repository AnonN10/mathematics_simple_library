#pragma once

#include <algorithm>
#include <type_traits>
#include <cstdint>
#include <iterator>
#include <functional>
#include <numeric>
#include <numbers>
#include <complex>
#include <array>

namespace Maths {
	
	using IndexType = std::size_t;
	
	template<typename T>
	struct is_complex_t : public std::false_type {};
	template<typename T>
	struct is_complex_t<std::complex<T>> : public std::true_type {};
	template<typename T>
	inline constexpr bool is_complex_v = is_complex_t<T>::value;
	
	template<typename T>
	constexpr bool is_power_of_two(T x) { return x && ((x & (x-1)) == 0); }

	template<IndexType M, IndexType N, typename Scalar = float, bool use_heap = false>
	struct Matrix
	{	
		//row major: row M, column N
		std::conditional_t<
			use_heap,
			std::vector<std::array<Scalar, N>>,
			std::array<std::array<Scalar, N>, M>
		> data;
		
		struct EmptyType {};
		using ElementIndexed = std::conditional_t<M == 1 || N == 1, Scalar, Matrix<M, 1, Scalar, use_heap>>;
		
		//constructors and setters
		
		Matrix() {
			if constexpr(use_heap) data.resize(M);
			identity();
		}
		Matrix(std::array<Scalar, M*N> flat_list) {
			if constexpr(use_heap) data.resize(M);
			set(flat_list);
		}
		
		Matrix<M, N, Scalar, use_heap>(Matrix<M, N, Scalar, use_heap> const&) = default;
		
		//flat list is stored in memory as is due to row major order
		void set(std::array<Scalar, M*N> flat_list) {
			for(IndexType m = 0; m < M; ++m)
				for(IndexType n = 0; n < N; ++n)
					data[m][n] = flat_list[N*m + n];
		}
		
		//builders
		
		void identity() {
			for(IndexType m = 0; m < M; ++m)
				for(IndexType n = 0; n < N; ++n)
					data[m][n] = m==n? Scalar(1) : Scalar(0);
		}
		
		void hadamard_identity() {
			for(IndexType m = 0; m < M; ++m)
				for(IndexType n = 0; n < N; ++n)
					data[m][n] = Scalar(1);
		}
		
		void DFT() {
			static_assert(N==M, "Operation undefined: DFT matrix must be square");
			static_assert(is_complex_v<Scalar>, "Operation undefined: DFT matrix must be complex");
			
			using VType = Scalar::value_type;
			
			constexpr Scalar i = Scalar(VType(0), VType(1));
			constexpr VType pi = std::numbers::pi_v<VType>;
			
			const VType norm = VType(1)/std::sqrt(VType(N));
			const Scalar omega = std::exp(VType(-2) * pi * i / Scalar(VType(N)));
			
			for(IndexType m = 0; m < M; ++m)
				for(IndexType n = 0; n < N; ++n)
					data[m][n] = std::pow(omega, VType(n*m));
		}
		
		void sylvester_walsh() {
			static_assert(N==M, "Operation undefined: Walsh matrices built using Sylvester's construction must be square");
			static_assert(is_power_of_two(N), "Operation undefined: Walsh matrices built using Sylvester's construction must have dimension which is a power of 2");
			
			if constexpr(N==1) {
				//Hadamard order 1
				data[0][0] = 1;
			} else if constexpr(N==2) {
				//Hadamard order 2
				set({
					1,  1,
					1, -1,
				});
			} else {
				Matrix<2, 2, Scalar, use_heap> H_2;
				Matrix<M/2, N/2, Scalar, use_heap> H_n;
				H_n.sylvester_walsh();
				H_2.sylvester_walsh();
				(*this) = H_2.kronecker_product(H_n);
			}
		}
		
		//getters

		std::array<Scalar, M*N> get() {
			std::array<Scalar, M*N> flat_list;
			for(IndexType m = 0; m < M; ++m)
				for(IndexType n = 0; n < N; ++n)
					flat_list[N*m + n] = data[m][n];
			return flat_list;
		}

		std::vector<Scalar> get_v() {
			std::vector<Scalar> flat_list(M*N);
			for(IndexType m = 0; m < M; ++m)
				for(IndexType n = 0; n < N; ++n)
					flat_list[N*m + n] = data[m][n];
			return flat_list;
		}
		
		Matrix<1, N, Scalar, use_heap> row(IndexType m) const {
			Matrix<1, N, Scalar, use_heap> result;
			for(IndexType n = 0; n < N; ++n)
				result[0][n] = data[m][n];
			return result;
		}
		
		Matrix<M, 1, Scalar, use_heap> column(IndexType n) const {
			Matrix<M, 1, Scalar, use_heap> result;
			for(IndexType m = 0; m < M; ++m)
				result.data[m][0] = data[m][n];
			return result;
		}

		template<IndexType N_first>
		Matrix<M, N-N_first, Scalar, use_heap> split_right() const {
			Matrix<M, N-N_first, Scalar, use_heap> result;
			for(IndexType m = 0; m < M; ++m)
				for(IndexType n = 0; n < N-N_first; ++n)
					result[m][n] = data[m][n+N_first];
			return result;
		}
		
		//operators
		
		template<typename T>
		explicit operator Matrix<M, N, T, use_heap>() const {
			Matrix<M, N, T, use_heap> result;
			for(IndexType m = 0; m < M; ++m)
				for(IndexType n = 0; n < N; ++n)
					result[m][n] = T(data[m][n]);
			return result;
		}
		
		ElementIndexed operator()(IndexType i) {
			//vector: index first column
			if constexpr(N == 1) return data[i][0];
			//covector: index first row
			else if constexpr(M == 1) return data[0][i];
			//matrix: index rows
			else return row(i);
		}
		
		Scalar operator()(IndexType row, IndexType column) {
			return data[row][column];
		}
		
		auto&& operator[](IndexType m) { return data[m]; }
		const auto operator[](IndexType m) const { return data[m]; }
		
		Matrix<M, N, Scalar, use_heap> operator-() const {
			Matrix<M, N, Scalar, use_heap> result;
			for(IndexType m = 0; m < M; ++m)
				for(IndexType n = 0; n < N; ++n)
					result.data[m][n] = -data[m][n];
			return result;
		}
		
		template<IndexType M_other, IndexType N_other, bool use_heap_other>
		Matrix<M, N_other, Scalar, use_heap || use_heap_other> operator*(const Matrix<M_other, N_other, Scalar, use_heap_other>& other) const {
			static_assert(N==M_other, "Operation undefined: for matrix multiplication the number of columns of the first must match the number of rows of the second");
			Matrix<M, N_other, Scalar, use_heap || use_heap_other> result;
			//matrix multiplication is essentially a collection of dot product permutations of
			//rows of the first and columnts of the second, therefore we can transpose the second
			//to perform just the dot products of row permutations
			Matrix<N_other, M_other, Scalar, use_heap_other> other_T = other.transpose();
			//for each element of the resulting matrix
			for(IndexType m = 0; m < M; ++m) {
				for(decltype(N_other) n = 0; n < N_other; ++n) {
					//dot product
					result.data[m][n] = std::transform_reduce(
											std::begin(this->data[m]), std::end(this->data[m]),
											std::begin(other_T.data[n]),
											Scalar(0), std::plus<>{}, std::multiplies<>{}
										);
				}
			}
			return result;
		}
		
		template<IndexType M_other, IndexType N_other, bool use_heap_other>
		Matrix<M, N_other, Scalar, use_heap> operator/(const Matrix<M_other, N_other, Scalar, use_heap_other>& other) const {
			return other.inverse()*(*this);
		}
		
		template<bool use_heap_other>
		Matrix<M, N, Scalar, use_heap> operator+(const Matrix<M, N, Scalar, use_heap_other>& other) const {
			Matrix<M, N, Scalar, use_heap> result;
			for(IndexType m = 0; m < M; ++m)
				for(IndexType n = 0; n < N; ++n)
					result.data[m][n] = data[m][n]+other[m][n];
			return result;
		}
		
		template<bool use_heap_other>
		Matrix<M, N, Scalar, use_heap> operator-(const Matrix<M, N, Scalar, use_heap_other>& other) const {
			return (*this)+(-other);
		}
		
		Matrix<M, N, Scalar, use_heap> operator*(Scalar scalar) const {
			Matrix<M, N, Scalar, use_heap> result;
			for(IndexType m = 0; m < M; ++m)
				for(IndexType n = 0; n < N; ++n)
					result.data[m][n] = data[m][n]*scalar;
			return result;
		}
		
		Matrix<M, N, Scalar, use_heap> operator/(Scalar scalar) const {
			Matrix<M, N, Scalar, use_heap> result;
			for(IndexType m = 0; m < M; ++m)
				for(IndexType n = 0; n < N; ++n)
					result.data[m][n] = data[m][n]/scalar;
			return result;
		}
		
		//methods

		template<IndexType M_other, IndexType N_other, bool use_heap_other>
		Matrix<M, N+N_other, Scalar, use_heap || use_heap_other> augment(const Matrix<M_other, N_other, Scalar, use_heap_other>& other) const {
			static_assert(M==M_other, "Operation undefined: matrix can only be augmented with a matrix with the same amount of rows");
			Matrix<M, N+N_other, Scalar, use_heap || use_heap_other> result;
			//copy current's columns
			for(IndexType m = 0; m < M; ++m)
				for(IndexType n = 0; n < N; ++n)
					result[m][n] = data[m][n];
			//copy other's columns
			for(IndexType m = 0; m < M_other; ++m)
				for(IndexType n = 0; n < N_other; ++n)
					result[m][N+n] = other[m][n];
			return result;
		}
		
		template<bool use_heap_other>
		Matrix<M, N, Scalar, use_heap> hadamard_product(const Matrix<M, N, Scalar, use_heap_other>& other) const {
			Matrix<M, N, Scalar, use_heap> result;
			for(IndexType m = 0; m < M; ++m)
				for(IndexType n = 0; n < N; ++n)
					result.data[m][n] = data[m][n]*other[m][n];
			return result;
		}
		
		template<IndexType M_other, IndexType N_other, bool use_heap_other>
		Matrix<M*M_other, N*N_other, Scalar, use_heap> kronecker_product(const Matrix<M_other, N_other, Scalar, use_heap_other>& other) const {
			Matrix<M*M_other, N*N_other, Scalar, use_heap> result;
			for(IndexType m = 0; m < M; ++m)
				for(IndexType n = 0; n < N; ++n)
					for(IndexType m_other = 0; m_other < M_other; ++m_other)
						for(IndexType n_other = 0; n_other < N_other; ++n_other)
							result.data[m*M_other+m_other][n*N_other+n_other] = data[m][n]*other[m_other][n_other];
			return result;
		}
		
		Scalar dot(const Matrix<M, N, Scalar, use_heap>& other) {
			static_assert(M==1||N==1, "Operation undefined: dot product is defined only for vectors and covectors");
			//vector
			if constexpr(N==1) return (this->transpose() * other)(0);
			//covector
			else return this->transpose().dot(other.transpose());
		}
		
		Matrix<N, M, Scalar, use_heap> transpose() const {
			Matrix<N, M, Scalar, use_heap> result;
			for(IndexType m = 0; m < M; ++m)
				for(IndexType n = 0; n < N; ++n)
					result.data[n][m] = this->data[m][n];
			return result;
		}

		//numerical stability improvements by Trolljanhorse
		Matrix<M, N, Scalar, use_heap> reduced_row_echelon_form() const {
			Matrix<M, N, Scalar, use_heap> result = *this;
			for(IndexType lead = 0; lead < M; ++lead) {
				Scalar divisor, multiplier;
				//1. Find largest entry in column `lead`
				IndexType pivot = lead;
				for (IndexType m = lead; m < M; ++m) {
					if (std::abs(result[pivot][lead]) < std::abs(result[m][lead])) {
						pivot = m;
					}
				}
				//2. Swap row `lead` with row of largest column
				for (IndexType n = 0; n < N; ++n) {
					std::swap(result[pivot][n], result[lead][n]);
				}

				for (IndexType m = 0; m < M; ++m) {
					divisor = result[lead][lead];
					multiplier = result[m][lead] / result[lead][lead];
					for (IndexType n = 0; n < N; ++n) {
						if (m == lead)
							result[m][n] /= divisor;
						else
							result[m][n] -= result[lead][n] * multiplier;
					}
				}
			}
			return result;
		}
		Matrix<M, N, Scalar, use_heap> rref() const { return reduced_row_echelon_form(); }
		
		Matrix<M-1, N-1, Scalar, use_heap> submatrix(IndexType m, IndexType n) const {
			Matrix<M-1, N-1, Scalar, use_heap> result;
			for(IndexType m_src = 0, m_dest = 0; m_src < M; ++m_src) {
				if(m_src == m) continue;
				for(IndexType n_src = 0, n_dest = 0; n_src < N; ++n_src) {
					if(n_src == n) continue;
					result.data[m_dest][n_dest] = this->data[m_src][n_src];
					++n_dest;
				}
				++m_dest;
			}
			return result;
		}
		
		Matrix<M, N, Scalar, use_heap> cofactor() const {
			Matrix<M, N, Scalar, use_heap> result;
			for(IndexType m = 0; m < M; ++m)
				for(IndexType n = 0; n < N; ++n)
					result.data[m][n] = Scalar((m+n)&1?-1:1)*submatrix(m,n).determinant();
			return result;
		}
		
		Matrix<M, N, Scalar, use_heap> adjugate() const {
			return cofactor().transpose();
		}
		
		Scalar determinant() const {
			static_assert(M==N, "Operation undefined: determinant is only defined for square matrices");
			if constexpr(M==1){
				return data[0][0];
			} else if constexpr(M==2) {
				return data[0][0]*data[1][1] - data[0][1]*data[1][0];
			} else {
				Scalar result = Scalar(0);
				for (int n = 0; n < N; n++)
					result += data[0][n] * Scalar(n&1?-1:1)*submatrix(0,n).determinant();
				return result;
			}
			return Scalar(0);
		}
		Scalar det() const { return determinant(); }
		
		Matrix<M, N, Scalar, use_heap> inverse_gauss_jordan() const {
			return augment(Matrix<M, N, Scalar, use_heap>()).rref().split_right<N>();
		}
		Matrix<M, N, Scalar, use_heap> inverse() const {
			return adjugate()/determinant();
		}
		Matrix<M, N, Scalar, use_heap> inv() const { return inverse(); }
		
		Matrix<M, N, Scalar, use_heap> conjugate() const {
			if constexpr(is_complex_v<Scalar>) {
				Matrix<M, N, Scalar, use_heap> result;
				for(IndexType m = 0; m < M; ++m)
					for(IndexType n = 0; n < N; ++n)
						result[m][n] = std::conj(data[m][n]);
				return result;
			} else return *this;
		}
		Matrix<M, N, Scalar, use_heap> conj() const { return conjugate(); }
		
		Matrix<N, N, Scalar, use_heap> gramian() const {
			return conjugate().transpose()*(*this);
		}
		Matrix<N, N, Scalar, use_heap> gram() const { return gramian(); }
		
		Scalar euclidean_norm() const {
			Scalar sum = Scalar(0);
			for(IndexType m = 0; m < M; ++m)
				for(IndexType n = 0; n < N; ++n)
					sum += data[m][n]*data[m][n];
			return std::sqrt(sum);
		}
		Scalar norm() const { return euclidean_norm(); }
		
		Matrix<M, N, Scalar, use_heap> euclidean_normalize() const {
			return (*this)/euclidean_norm();
		}
		Matrix<M, N, Scalar, use_heap> normalize() const { return euclidean_normalize(); }
	};

	//column-vector
	template<IndexType M, typename Scalar = float, bool use_heap = false>
	using Vector = Matrix<M, 1, Scalar, use_heap>;

	//row-vector
	template<IndexType N, typename Scalar = float, bool use_heap = false>
	using Covector = Matrix<1, N, Scalar, use_heap>;

} // namespace Maths