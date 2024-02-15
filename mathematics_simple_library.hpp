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

	template<IndexType M, IndexType N, typename Scalar = float>
	struct Matrix
	{	
		//row major: row M, column N
		Scalar data[M][N];
		
		struct EmptyType {};
		using ElementIndexed = std::conditional_t<M == 1 || N == 1, Scalar, Matrix<M, 1, Scalar>>;
		
		//constructors and setters
		
		Matrix() {
			identity();
		}
		Matrix(std::array<Scalar, M*N> flat_list) {
			set(flat_list);
		}
		
		Matrix<M, N, Scalar>(Matrix<M, N, Scalar> const&) = default;
		
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
					data[m][n] = m==n? 1.0f : 0.0f;
		}
		
		void DFT() {
			static_assert(N==M, "Operation undefined: DFT matrix must be square");
			static_assert(is_complex_v<Scalar>, "Operation undefined: DFT matrix must be complex");
			
			using VType = Scalar::value_type;
			
			constexpr Scalar i = Scalar(VType(0.0), VType(1.0));
			constexpr VType pi = std::numbers::pi_v<VType>;
			
			const VType norm = VType(1)/std::sqrt(VType(N));
			const Scalar omega = std::exp(VType(-2.0) * pi * i / Scalar(VType(N)));
			
			for(IndexType m = 0; m < M; ++m)
				for(IndexType n = 0; n < N; ++n)
					data[m][n] = std::pow(omega, VType(n*m));
		}
		
		//getters
		
		Matrix<1, N, Scalar> row(IndexType m) const {
			Matrix<1, N, Scalar> result;
			for(IndexType n = 0; n < N; ++n)
				result[0][n] = data[m][n];
			return result;
		}
		
		Matrix<M, 1, Scalar> column(IndexType n) const {
			Matrix<M, 1, Scalar> result;
			for(IndexType m = 0; m < M; ++m)
				result.data[m][0] = data[m][n];
			return result;
		}
		
		//operators
		
		template<typename T>
		explicit operator Matrix<M, N, T>() const {
			Matrix<M, N, T> result;
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
		
		Scalar* operator[](IndexType m) { return data[m]; }
		const Scalar* operator[](IndexType m) const { return data[m]; }
		
		Matrix<M, N, Scalar> operator-() const {
			Matrix<M, N, Scalar> result;
			for(IndexType m = 0; m < M; ++m)
				for(IndexType n = 0; n < N; ++n)
					result.data[m][n] = -data[m][n];
			return result;
		}
		
		template<IndexType M_other, IndexType N_other>
		Matrix<M, N_other, Scalar> operator*(const Matrix<M_other, N_other, Scalar>& other) const {
			static_assert(N==M_other, "Operation undefined: for matrix multiplication the number of columns of the first must match the number of rows of the second");
			Matrix<M, N_other, Scalar> result;
			//matrix multiplication is essentially a collection of dot product permutations of
			//rows of the first and columnts of the second, therefore we can transpose the second
			//to perform just the dot products of row permutations
			Matrix<N_other, M_other, Scalar> other_T = other.transpose();
			//for each element of the resulting matrix
			for(IndexType m = 0; m < M; ++m) {
				for(decltype(N_other) n = 0; n < N_other; ++n) {
					//dot product
					result.data[m][n] = std::transform_reduce(
											std::begin(this->data[m]), std::end(this->data[m]),
											std::begin(other_T.data[n]),
											Scalar(0.0), std::plus<>{}, std::multiplies<>{}
										);
				}
			}
			return result;
		}
		
		template<IndexType M_other, IndexType N_other>
		Matrix<M, N_other, Scalar> operator/(const Matrix<M_other, N_other, Scalar>& other) const {
			return (*this)/other.inverse();
		}
		
		Matrix<M, N, Scalar> operator+(const Matrix<M, N, Scalar>& other) const {
			Matrix<M, N, Scalar> result;
			for(IndexType m = 0; m < M; ++m)
				for(IndexType n = 0; n < N; ++n)
					result.data[m][n] = data[m][n]+other[m][n];
			return result;
		}
		
		Matrix<M, N, Scalar> operator-(const Matrix<M, N, Scalar>& other) const {
			return (*this)+(-other);
		}
		
		Matrix<M, N, Scalar> operator*(Scalar scalar) const {
			Matrix<M, N, Scalar> result;
			for(IndexType m = 0; m < M; ++m)
				for(IndexType n = 0; n < N; ++n)
					result.data[m][n] = data[m][n]*scalar;
			return result;
		}
		
		Matrix<M, N, Scalar> operator/(Scalar scalar) const {
			Matrix<M, N, Scalar> result;
			for(IndexType m = 0; m < M; ++m)
				for(IndexType n = 0; n < N; ++n)
					result.data[m][n] = data[m][n]/scalar;
			return result;
		}
		
		//methods
		
		Matrix<M, N, Scalar> hadamard_product(const Matrix<M, N, Scalar>& other) const {
			Matrix<M, N, Scalar> result;
			for(IndexType m = 0; m < M; ++m)
				for(IndexType n = 0; n < N; ++n)
					result.data[m][n] = data[m][n]*other[m][n];
			return result;
		}
		
		template<IndexType M_other, IndexType N_other>
		Matrix<M*M_other, N*N_other, Scalar> kronecker_product(const Matrix<M_other, N_other, Scalar>& other) const {
			Matrix<M*M_other, N*N_other, Scalar> result;
			for(IndexType m = 0; m < M; ++m)
				for(IndexType n = 0; n < N; ++n)
					for(IndexType m_other = 0; m_other < M_other; ++m_other)
						for(IndexType n_other = 0; n_other < N_other; ++n_other)
							result.data[m*M_other+m_other][n*N_other+n_other] = data[m][n]*other[m_other][n_other];
			return result;
		}
		
		Scalar dot(const Matrix<M, N>& other) {
			static_assert(M==1||N==1, "Operation undefined: dot product is defined only for vectors and covectors");
			//vector
			if constexpr(N==1) return (this->transpose() * other)(0);
			//covector
			else return this->transpose().dot(other.transpose());
		}
		
		Matrix<N, M, Scalar> transpose() const {
			Matrix<N, M, Scalar> result;
			for(IndexType m = 0; m < M; ++m)
				for(IndexType n = 0; n < N; ++n)
					result.data[n][m] = this->data[m][n];
			return result;
		}
		
		Matrix<M-1, N-1, Scalar> submatrix(IndexType m, IndexType n) const {
			Matrix<M-1, N-1, Scalar> result;
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
		
		Matrix<M, N, Scalar> cofactor() const {
			Matrix<M, N, Scalar> result;
			for(IndexType m = 0; m < M; ++m)
				for(IndexType n = 0; n < N; ++n)
					result.data[m][n] = Scalar((m+n)&1?-1:1)*submatrix(m,n).determinant();
			return result;
		}
		
		Matrix<M, N, Scalar> adjugate() const {
			return cofactor().transpose();
		}
		
		Scalar determinant() const {
			static_assert(M==N, "Operation undefined: determinant is only defined for square matrices");
			if constexpr(M==1){
				return data[0][0];
			} else if constexpr(M==2) {
				return data[0][0]*data[1][1] - data[0][1]*data[1][0];
			} else {
				Scalar result = 0.0f;
				for (int n = 0; n < N; n++)
					result += data[0][n] * Scalar(n&1?-1:1)*submatrix(0,n).determinant();
				return result;
			}
			return 0.0f;
		}
		Scalar det() const { return determinant(); }
		
		Matrix<M, N, Scalar> inverse() const {
			return adjugate()/determinant();
		}
		Matrix<M, N, Scalar> inv() const { return inverse(); }
		
		Matrix<M, N, Scalar> conjugate() const {
			if constexpr(is_complex_v<Scalar>) {
				Matrix<M, N, Scalar> result;
				for(IndexType m = 0; m < M; ++m)
					for(IndexType n = 0; n < N; ++n)
						result[m][n] = std::conj(data[m][n]);
				return result;
			} else return *this;
		}
		Matrix<M, N, Scalar> conj() const { return conjugate(); }
		
		Matrix<N, N, Scalar> gramian() const {
			return conjugate().transpose()*(*this);
		}
		Matrix<N, N, Scalar> gram() const { return gramian(); }
		
		Scalar euclidean_norm() const {
			Scalar sum = Scalar(0.0);
			for(IndexType m = 0; m < M; ++m)
				for(IndexType n = 0; n < N; ++n)
					sum += data[m][n]*data[m][n];
			return std::sqrt(sum);
		}
		Scalar norm() const { return euclidean_norm(); }
		
		Matrix<M, N, Scalar> euclidean_normalize() const {
			return (*this)/euclidean_norm();
		}
		Matrix<M, N, Scalar> normalize() const { return euclidean_normalize(); }
	};

	//column-vector
	template<IndexType M, typename Scalar = float>
	using Vector = Matrix<M, 1, Scalar>;

	//row-vector
	template<IndexType N, typename Scalar = float>
	using Covector = Matrix<1, N, Scalar>;

} // namespace Maths