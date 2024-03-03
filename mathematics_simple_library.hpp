#pragma once

#include <cassert>
#include <cstdlib>
#include <algorithm>
#include <type_traits>
#include <cstdint>
#include <iterator>
#include <functional>
#include <numeric>
#include <numbers>
#include <complex>
#include <array>
#include <iterator>
#include <cstddef>
#include <span>

namespace Maths {
	
	using IndexType = std::size_t;
	
	template<typename T>
	struct is_complex_t : public std::false_type {};
	template<typename T>
	struct is_complex_t<std::complex<T>> : public std::true_type {};
	template<typename T>
	inline constexpr bool is_complex_v = is_complex_t<T>::value;

	template <class T, class = void> struct value_type { using type = T; };
	template <class T> struct value_type<T, std::void_t<typename T::value_type>> { using type = typename T::value_type; };
	template <class T> using value_type_t = typename value_type<T>::type;
	
	template<typename T>
	constexpr bool is_power_of_two(T x) { return x && ((x & (x-1)) == 0); }

	//NOTE: may return negative zero
	template<typename T>
	constexpr T euclidean_remainder(T a, T b) {
		assert(b != T(0));
		T result;
		if constexpr(std::is_integral_v<T>) {
			result = a % b;
		} else {
			result = std::fmod(a,b);
		}
		if constexpr(!std::is_unsigned_v<T>) b = std::abs(b);
		return result >= T(0) ? result : result + b;
	}
	template<typename T>
	constexpr T euclidean_modulo(T a, T b) { return euclidean_remainder(a, b); }
	template<typename T>
	constexpr T eucmod(T a, T b) { return euclidean_remainder(a, b); }

	template<IndexType M, IndexType N, typename Field = float, bool use_heap = false>
	struct Matrix
	{	
		//row major data layout: row M, column N
		std::conditional_t<
			use_heap,
			std::vector<Field>,
			std::array<Field, M*N>
		> data;
		
		struct EmptyType {};
		using ElementIndexed = std::conditional_t<
			M == 1 || N == 1,
			Field,
			Matrix<M, 1, Field, use_heap>
		>;
		using ValueType = value_type_t<Field>;
		
		//constructors and setters
		
		Matrix() {
			if constexpr(use_heap) data.resize(M*N);
			if constexpr(std::is_constructible_v<Field, int>) identity();
		}
		Matrix(std::array<Field, M*N> flat_list) {
			if constexpr(use_heap) data.resize(M*N);
			set(flat_list);
		}
		
		Matrix(Matrix<M, N, Field, use_heap> const&) = default;
		
		//flat list is stored in memory as is due to row major order
		void set(const std::array<Field, M*N>& flat_list) {
            set(std::span { flat_list });
        }

		void set(std::span<const Field, M*N> flat_list) {
			for(IndexType m = 0; m < M; ++m)
				for(IndexType n = 0; n < N; ++n)
					(*this)[m, n] = flat_list[N*m + n];
		}

		void set(std::span<const Field> v) {
			assert(v.size() == M*N);
			set(std::span<const Field, M*N> { v });
		}

		//builders
		
		void identity() {
			static_assert(std::is_constructible_v<Field, int>, "Invalid operation: matrix element type is not constructible from int");
			for(IndexType m = 0; m < M; ++m)
				for(IndexType n = 0; n < N; ++n)
					(*this)[m, n] = m==n? Field(1) : Field(0);
		}
		
		void identity_hadamard() {
			static_assert(std::is_constructible_v<Field, int>, "Invalid operation: matrix element type is not constructible from int");
			for(IndexType m = 0; m < M; ++m)
				for(IndexType n = 0; n < N; ++n)
					(*this)[m, n] = Field(1);
		}

		void identity_additive() {
			static_assert(std::is_constructible_v<Field, int>, "Invalid operation: matrix element type is not constructible from int");
			for(IndexType m = 0; m < M; ++m)
				for(IndexType n = 0; n < N; ++n)
					(*this)[m, n] = Field(0);
		}
		void zero() { identity_additive(); }
		
		void DFT() {
			static_assert(N==M, "Operation undefined: DFT matrix must be square");
			static_assert(is_complex_v<Field>, "Operation undefined: DFT matrix must be complex");
			
			constexpr Field i = Field(ValueType(0), ValueType(1));
			constexpr ValueType pi = std::numbers::pi_v<ValueType>;
			
			const ValueType norm = ValueType(1)/std::sqrt(ValueType(N));
			const Field omega = std::exp(ValueType(-2) * pi * i / Field(ValueType(N)));
			
			for(IndexType m = 0; m < M; ++m)
				for(IndexType n = 0; n < N; ++n)
					(*this)[m, n] = std::pow(omega, ValueType(n*m))*norm;
		}
		
		void sylvester_walsh() {
			static_assert(N==M, "Operation undefined: Walsh matrices built using Sylvester's construction must be square");
			static_assert(is_power_of_two(N), "Operation undefined: Walsh matrices built using Sylvester's construction must have dimension which is a power of 2");
			
			if constexpr(N==1) {
				//Hadamard order 1
				(*this)[0] = 1;
			} else if constexpr(N==2) {
				//Hadamard order 2
				set({
					1,  1,
					1, -1,
				});
			} else {
				Matrix<2, 2, Field, use_heap> H_2;
				Matrix<M/2, N/2, Field, use_heap> H_n;
				H_n.sylvester_walsh();
				H_2.sylvester_walsh();
				(*this) = H_2.kronecker_product(H_n);
			}
		}
		
		//getters and iterators

		auto begin() { return data.begin(); }
		auto end() { return data.end(); }
		const auto begin() const { return data.begin(); }
		const auto end() const { return data.end(); }

		std::array<Field, M*N> get() {
			std::array<Field, M*N> flat_list;
			for(IndexType m = 0; m < M; ++m)
				for(IndexType n = 0; n < N; ++n)
					flat_list[N*m + n] = (*this)[m, n];
			return flat_list;
		}

		std::vector<Field> get_v() {
			std::vector<Field> flat_list(data.begin(), data.end());
			return flat_list;
		}
		
		Matrix<1, N, Field, use_heap> row(IndexType m) const {
			Matrix<1, N, Field, use_heap> result;
			for(IndexType n = 0; n < N; ++n)
				result[0, n] = (*this)[m, n];
			return result;
		}
		
		Matrix<M, 1, Field, use_heap> column(IndexType n) const {
			Matrix<M, 1, Field, use_heap> result;
			for(IndexType m = 0; m < M; ++m)
				result[m, 0] = (*this)[m, n];
			return result;
		}

		template<IndexType N_first>
		Matrix<M, N-N_first, Field, use_heap> split_right() const {
			Matrix<M, N-N_first, Field, use_heap> result;
			for(IndexType m = 0; m < M; ++m)
				for(IndexType n = 0; n < N-N_first; ++n)
					result[m, n] = (*this)[m, n+N_first];
			return result;
		}
		
		//operators
		
		template<typename T>
		explicit operator Matrix<M, N, T, use_heap>() const {
			Matrix<M, N, T, use_heap> result;
			for(IndexType m = 0; m < M; ++m)
				for(IndexType n = 0; n < N; ++n)
					result[m, n] = T((*this)[m, n]);
			return result;
		}
		
		ElementIndexed operator()(IndexType i) {
			//vector: index first column
			if constexpr(N == 1) return (*this)[i, 0];
			//covector: index first row
			else if constexpr(M == 1) return (*this)[0, i];
			//matrix: index rows
			else return row(i);
		}
		
		Field operator()(IndexType row, IndexType column) {
			return (*this)[row, column];
		}
		
		auto&& operator[](IndexType i) { return data[i]; }
		const auto operator[](IndexType i) const { return data[i]; }
		auto&& operator[](IndexType m, IndexType n) { return data[N*m+n]; }
		const auto operator[](IndexType m, IndexType n) const { return data[N*m+n]; }
		
		Matrix<M, N, Field, use_heap> operator-() const {
			Matrix<M, N, Field, use_heap> result;
			for(IndexType m = 0; m < M; ++m)
				for(IndexType n = 0; n < N; ++n)
					result[m, n] = -(*this)[m, n];
			return result;
		}
		
		template<IndexType M_other, IndexType N_other, bool use_heap_other>
		Matrix<M, N_other, Field, use_heap || use_heap_other> operator*(const Matrix<M_other, N_other, Field, use_heap_other>& other) const {
			static_assert(N==M_other, "Operation undefined: for matrix multiplication the number of columns of the first must match the number of rows of the second");
			Matrix<M, N_other, Field, use_heap || use_heap_other> result;
			//matrix multiplication is essentially a collection of dot product permutations of
			//rows of the first and columnts of the second, therefore we can transpose the second
			//to perform just the dot products of row permutations
			auto other_T = other.transpose();
			//for each element of the resulting matrix
			for(IndexType m = 0; m < M; ++m) {
				for(decltype(N_other) n = 0; n < N_other; ++n) {
					//dot product
					result[m, n] = std::transform_reduce(
						this->begin()+N*m, this->begin()+N*(m+1),
						other_T.begin()+M_other*n,
						Field(0), std::plus<>{}, std::multiplies<>{}
					);
				}
			}
			return result;
		}
		
		template<IndexType M_other, IndexType N_other, bool use_heap_other>
		Matrix<M, N_other, Field, use_heap> operator/(const Matrix<M_other, N_other, Field, use_heap_other>& other) const {
			return other.inverse()*(*this);
		}
		
		template<bool use_heap_other>
		Matrix<M, N, Field, use_heap> operator+(const Matrix<M, N, Field, use_heap_other>& other) const {
			Matrix<M, N, Field, use_heap> result;
			for(IndexType m = 0; m < M; ++m)
				for(IndexType n = 0; n < N; ++n)
					result[m, n] = (*this)[m, n]+other[m, n];
			return result;
		}
		
		template<bool use_heap_other>
		Matrix<M, N, Field, use_heap> operator-(const Matrix<M, N, Field, use_heap_other>& other) const {
			return (*this)+(-other);
		}
		
		Matrix<M, N, Field, use_heap> operator*(Field scalar) const {
			Matrix<M, N, Field, use_heap> result;
			for(IndexType m = 0; m < M; ++m)
				for(IndexType n = 0; n < N; ++n)
					result[m, n] = (*this)[m, n]*scalar;
			return result;
		}
		
		Matrix<M, N, Field, use_heap> operator/(Field scalar) const {
			Matrix<M, N, Field, use_heap> result;
			for(IndexType m = 0; m < M; ++m)
				for(IndexType n = 0; n < N; ++n)
					result[m, n] = (*this)[m, n]/scalar;
			return result;
		}

		Matrix<M, N, Field, use_heap> operator+(Field scalar) const {
			Matrix<M, N, Field, use_heap> result;
			for(IndexType m = 0; m < M; ++m)
				for(IndexType n = 0; n < N; ++n)
					result[m, n] = (*this)[m, n]+scalar;
			return result;
		}
		
		Matrix<M, N, Field, use_heap> operator-(Field scalar) const {
			Matrix<M, N, Field, use_heap> result;
			for(IndexType m = 0; m < M; ++m)
				for(IndexType n = 0; n < N; ++n)
					result[m, n] = (*this)[m, n]-scalar;
			return result;
		}
		
		//methods

		template<IndexType M_other, IndexType N_other, bool use_heap_other>
		Matrix<M, N+N_other, Field, use_heap || use_heap_other> augment(const Matrix<M_other, N_other, Field, use_heap_other>& other) const {
			static_assert(M==M_other, "Operation undefined: matrix can only be augmented with a matrix with the same amount of rows");
			Matrix<M, N+N_other, Field, use_heap || use_heap_other> result;
			//copy current's columns
			for(IndexType m = 0; m < M; ++m)
				for(IndexType n = 0; n < N; ++n)
					result[m, n] = (*this)[m, n];
			//copy other's columns
			for(IndexType m = 0; m < M_other; ++m)
				for(IndexType n = 0; n < N_other; ++n)
					result[m, N+n] = other[m, n];
			return result;
		}
		
		template<bool use_heap_other>
		Matrix<M, N, Field, use_heap> hadamard_product(const Matrix<M, N, Field, use_heap_other>& other) const {
			Matrix<M, N, Field, use_heap> result;
			for(IndexType m = 0; m < M; ++m)
				for(IndexType n = 0; n < N; ++n)
					result[m, n] = (*this)[m, n]*other[m, n];
			return result;
		}
		
		template<IndexType M_other, IndexType N_other, bool use_heap_other>
		Matrix<M*M_other, N*N_other, Field, use_heap> kronecker_product(const Matrix<M_other, N_other, Field, use_heap_other>& other) const {
			Matrix<M*M_other, N*N_other, Field, use_heap> result;
			for(IndexType m = 0; m < M; ++m)
				for(IndexType n = 0; n < N; ++n)
					for(IndexType m_other = 0; m_other < M_other; ++m_other)
						for(IndexType n_other = 0; n_other < N_other; ++n_other)
							result[m*M_other+m_other, n*N_other+n_other] = (*this)[m, n]*other[m_other, n_other];
			return result;
		}
		
		Field dot(const Matrix<M, N, Field, use_heap>& other) {
			static_assert(M==1||N==1, "Operation undefined: dot product is defined only for vectors and covectors");
			//vector
			if constexpr(N==1) return (this->transpose() * other)(0);
			//covector
			else return this->transpose().dot(other.transpose());
		}
		
		Matrix<N, M, Field, use_heap> transpose() const {
			Matrix<N, M, Field, use_heap> result;
			for(IndexType m = 0; m < M; ++m)
				for(IndexType n = 0; n < N; ++n)
					result[n, m] = (*this)[m, n];
			return result;
		}

		Matrix<N, M, Field, use_heap> transpose_hermitian() const {
			return conjugate().transpose();
		}

		//numerical stability improvements by Trolljanhorse
		Matrix<M, N, Field, use_heap> reduced_row_echelon_form() const {
			using std::abs;
			
			Matrix<M, N, Field, use_heap> result = *this;
			for(IndexType lead = 0; lead < M; ++lead) {
				Field divisor, multiplier;
				//find largest entry in column `lead`
				IndexType pivot = lead;
				for (IndexType m = lead; m < M; ++m)
					if (abs(result[pivot, lead]) < abs(result[m, lead]))
						pivot = m;
				//swap row `lead` with row of largest column
				if(pivot != lead)
					for (IndexType n = 0; n < N; ++n)
						std::swap(result[pivot, n], result[lead, n]);

				for (IndexType m = 0; m < M; ++m) {
					divisor = result[lead, lead];
					if(divisor == Field(0)) continue;

					multiplier = result[m, lead] / divisor;
					for (IndexType n = 0; n < N; ++n)
						if (m == lead)
							result[m, n] /= divisor;
						else
							result[m, n] -= result[lead, n] * multiplier;
				}
			}
			return result;
		}
		Matrix<M, N, Field, use_heap> rref() const { return reduced_row_echelon_form(); }
		
		Matrix<M-1, N-1, Field, use_heap> submatrix(IndexType m, IndexType n) const {
			Matrix<M-1, N-1, Field, use_heap> result;
			for(IndexType m_src = 0, m_dest = 0; m_src < M; ++m_src) {
				if(m_src == m) continue;
				for(IndexType n_src = 0, n_dest = 0; n_src < N; ++n_src) {
					if(n_src == n) continue;
					result[m_dest, n_dest] = (*this)[m_src, n_src];
					++n_dest;
				}
				++m_dest;
			}
			return result;
		}
		
		Matrix<M, N, Field, use_heap> cofactor() const {
			Matrix<M, N, Field, use_heap> result;
			for(IndexType m = 0; m < M; ++m)
				for(IndexType n = 0; n < N; ++n)
					result[m, n] = Field((m+n)&1?-1:1)*submatrix(m,n).determinant();
			return result;
		}
		
		Matrix<M, N, Field, use_heap> adjugate() const {
			return cofactor().transpose();
		}
		
		Field determinant() const {
			static_assert(M==N, "Operation undefined: determinant is only defined for square matrices");
			if constexpr(M==1){
				return (*this)[0, 0];
			} else if constexpr(M==2) {
				return (*this)[0, 0]*(*this)[1, 1] - (*this)[0, 1]*(*this)[1, 0];
			} else {
				Field result = Field(0);
				for (int n = 0; n < N; n++)
					result += (*this)[0, n] * Field(n&1?-1:1)*submatrix(0,n).determinant();
				return result;
			}
			return Field(0);
		}
		Field det() const { return determinant(); }
		
		Matrix<M, N, Field, use_heap> inverse_gauss_jordan() const {
			return augment(Matrix<M, N, Field, use_heap>()).rref().template split_right<N>();
		}
		Matrix<M, N, Field, use_heap> inverse() const {
			return adjugate()/determinant();
		}
		Matrix<M, N, Field, use_heap> inv() const { return inverse(); }
		
		Matrix<M, N, Field, use_heap> conjugate() const {
			if constexpr(is_complex_v<Field>) {
				Matrix<M, N, Field, use_heap> result;
				for(IndexType m = 0; m < M; ++m)
					for(IndexType n = 0; n < N; ++n)
						result[m, n] = std::conj((*this)[m, n]);
				return result;
			} else return *this;
		}
		Matrix<M, N, Field, use_heap> conj() const { return conjugate(); }
		
		Matrix<N, N, Field, use_heap> gramian() const {
			return conjugate().transpose()*(*this);
		}
		Matrix<N, N, Field, use_heap> gram() const { return gramian(); }

		Field max() const {
			Field maximum = Field(0);
			for(IndexType m = 0; m < M; ++m)
				for(IndexType n = 0; n < N; ++n) {
					Field value = (*this)[m, n];
					if((m==0 && n==0) || maximum < value)
						maximum = value;
				}
			return maximum;
		}

		Field min() const {
			Field minimum = Field(0);
			for(IndexType m = 0; m < M; ++m)
				for(IndexType n = 0; n < N; ++n) {
					Field value = (*this)[m, n];
					if((m==0 && n==0) || minimum > value)
						minimum = value;
				}
			return minimum;
		}

		ValueType norm_max() const {
			using std::abs;
			ValueType maximum = ValueType(0);
			for(IndexType m = 0; m < M; ++m)
				for(IndexType n = 0; n < N; ++n) {
					ValueType magnitude = abs((*this)[m, n]);
					if((m==0 && n==0) || maximum < magnitude)
						maximum = magnitude;
				}
			return maximum;
		}

		ValueType norm_min() const {
			using std::abs;
			ValueType minimum = ValueType(0);
			for(IndexType m = 0; m < M; ++m)
				for(IndexType n = 0; n < N; ++n) {
					ValueType magnitude = abs((*this)[m, n]);
					if((m==0 && n==0) || minimum > magnitude)
						minimum = magnitude;
				}
			return minimum;
		}
		
		ValueType norm_frobenius() const {
			using std::sqrt;
			ValueType sum = ValueType(0);
			for(IndexType m = 0; m < M; ++m)
				for(IndexType n = 0; n < N; ++n)
					sum += (*this)[m, n]*(*this)[m, n];
			return sqrt(sum);
		}
		ValueType norm_euclidean() const { return norm_frobenius(); }
		ValueType norm() const { return norm_frobenius(); }
		
		Matrix<M, N, Field, use_heap> normalize_frobenius() const {
			return (*this)/norm_frobenius();
		}
		Matrix<M, N, Field, use_heap> normalize_euclidean() const { return normalize_frobenius(); }
		Matrix<M, N, Field, use_heap> normalize() const { return normalize_frobenius(); }

		Matrix<M, N, Field, use_heap> normalize_min() const { return (*this)/Field(norm_min()); }
		Matrix<M, N, Field, use_heap> normalize_max() const { return (*this)/Field(norm_max()); }
		Matrix<M, N, Field, use_heap> normalize_minmax() const {
			auto minimum = min();
			auto maximum = max();
			return ((*this) - Field(minimum))/(Field(maximum - minimum));
		}
	};

	//column-vector
	template<IndexType M, typename Field = float, bool use_heap = false>
	using Vector = Matrix<M, 1, Field, use_heap>;

	//row-vector
	template<IndexType N, typename Field = float, bool use_heap = false>
	using Covector = Matrix<1, N, Field, use_heap>;

} // namespace Maths
